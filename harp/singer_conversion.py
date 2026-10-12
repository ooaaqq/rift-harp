"""Resident target-singer conversion and a single-output CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import HARPConfig
from .f0 import FCPEFrontend
from .feature_contract import FeatureContract, validate_feature_contract
from .flow import HARPFlow
from .flow_transform import FlowTransform
from .model import HARPCore
from .performance import configure_cuda, resolve_device
from .vocoder import (
    load_pc_nsf,
    prepare_pc_nsf_harmonic_source,
    synthesize_pc_nsf,
)


class FrozenContentEncoder(nn.Module):
    """Pinned ContentVec backbone used by the foundation feature pipeline."""

    def __init__(self, backbone: nn.Module, content_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        if int(backbone.config.hidden_size) != content_dim:
            raise ValueError("ContentVec hidden size differs from HARP content_dim")
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.eval()

    @classmethod
    def load(cls, path: Path, content_dim: int) -> FrozenContentEncoder:
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        return cls(backbone, content_dim)

    def train(self, mode: bool = True) -> FrozenContentEncoder:
        super().train(False)
        return self

    def forward(self, waveform: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        output: Any = self.backbone(
            input_values=waveform,
            attention_mask=mask.long(),
            return_dict=True,
        )
        hidden = output.last_hidden_state
        mask_builder = getattr(
            self.backbone, "_get_feature_vector_attention_mask", None
        )
        if mask_builder is not None:
            hidden_mask = mask_builder(hidden.shape[1], mask.long()).bool()
        else:
            hidden_mask = (
                F.interpolate(
                    mask.float().unsqueeze(1), size=hidden.shape[1], mode="nearest"
                )
                .squeeze(1)
                .bool()
            )
        return hidden, hidden_mask


@torch.inference_mode()
def encode_content(
    encoder: FrozenContentEncoder,
    waveform: Tensor,
    sample_rate: int,
    device: torch.device,
    *,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 1.0,
    phase_shift_seconds: float = 0.01,
) -> Tensor:
    shift = round(phase_shift_seconds * sample_rate)
    shifted = torch.zeros_like(waveform)
    shifted[:-shift] = waveform[shift:]
    valid = torch.ones_like(waveform, dtype=torch.bool)
    shifted_valid = valid.clone()
    shifted_valid[-shift:] = False
    phases = []
    for values, value_mask in ((waveform, valid), (shifted, shifted_valid)):
        phases.append(
            _encode_content_phase(
                encoder,
                values,
                value_mask,
                sample_rate,
                device,
                chunk_seconds,
                overlap_seconds,
            )
        )
    frames = min(phases[0].shape[0], phases[1].shape[0])
    if frames < 2:
        raise ValueError("ContentVec extraction produced too few frames")
    return torch.stack(
        (phases[0][: frames - 1], phases[1][: frames - 1]), dim=1
    ).reshape(2 * (frames - 1), -1)


def _encode_content_phase(
    encoder: FrozenContentEncoder,
    waveform: Tensor,
    valid: Tensor,
    sample_rate: int,
    device: torch.device,
    chunk_seconds: float,
    overlap_seconds: float,
) -> Tensor:
    chunk = round(chunk_seconds * sample_rate)
    overlap = round(overlap_seconds * sample_rate)
    stride = chunk - overlap
    if stride <= 0:
        raise ValueError("ContentVec chunk must exceed overlap")
    pieces = []
    start = 0
    while start < waveform.numel():
        stop = min(waveform.numel(), start + chunk)
        segment = waveform[start:stop].to(device)[None]
        segment_mask = valid[start:stop].to(device)[None]
        output, output_mask = encoder(segment, segment_mask)
        count = int(output_mask[0].sum())
        content = output[0, :count]
        ratio = content.shape[0] / segment.shape[1]
        left = round(overlap / 2 * ratio) if start else 0
        right_trim = round(overlap / 2 * ratio) if stop < waveform.numel() else 0
        right = content.shape[0] - right_trim
        pieces.append(content[left:right].float().cpu())
        if stop == waveform.numel():
            break
        start += stride
    return torch.cat(pieces)


class FeatureExtractor:
    """Keep ContentVec and FCPE loaded for a queue of songs on one device."""

    def __init__(self, content_model: Path, config: HARPConfig, device: torch.device):
        self.config = config
        self.device = device
        self.encoder = FrozenContentEncoder.load(
            content_model, config.model.content_dim
        )
        self.encoder.to(device).eval()
        self.pitch_frontend = FCPEFrontend.load(device)

    @torch.inference_mode()
    def extract(self, input_path: Path) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        config, device = self.config, self.device
        import soundfile as sf
        import torchaudio.functional as AF

        audio, source_rate = sf.read(input_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio).mean(dim=1)
        if source_rate != config.feature.sample_rate:
            target_samples = round(
                Fraction(len(audio), source_rate) * config.feature.sample_rate
            )
            waveform = AF.resample(waveform, source_rate, config.feature.sample_rate)[
                :target_samples
            ]
        if not torch.isfinite(waveform).all():
            raise ValueError("input contains non-finite samples")
        expected_samples = waveform.numel()
        hop = config.feature.hop_length
        frames = (expected_samples + hop - 1) // hop
        if frames <= 0:
            raise ValueError("input is shorter than one HARP frame")

        # Complete the last feature frame with reflected input context. Synthesis
        # covers the original tail; only context beyond the source is trimmed.
        context = frames * hop - expected_samples
        if context:
            waveform = F.pad(waveform[None, None], (0, context), mode="reflect")[0, 0]
        content_waveform = AF.resample(waveform, config.feature.sample_rate, 16_000)
        encoder = self.encoder
        content = encode_content(encoder, content_waveform, 16_000, device)
        content = _resize_matrix(content, frames)

        pitch_frontend = self.pitch_frontend
        f0_track = pitch_frontend.extract_compatible(
            waveform.clamp(-1, 1),
            config.feature.sample_rate,
            target_sample_rate=config.feature.sample_rate,
            target_hop_length=config.feature.hop_length,
            threshold=0.006,
            f0_min=config.harmonic.f0_min,
            f0_max=config.harmonic.f0_max,
        )
        f0 = f0_track.f0_hz
        if f0.numel() != frames:
            raise RuntimeError("canonical F0 track differs from HARP feature frames")

        window = config.feature.win_length
        pad = (window - config.feature.hop_length) // 2
        padded = F.pad(waveform[None, None], (pad, pad), mode="reflect")[0, 0]
        rms = (
            padded.unfold(0, window, config.feature.hop_length).square().mean(-1).sqrt()
        )
        rms = _resize_vector(rms, frames)
        return waveform, content, f0, rms, expected_samples


def _resize_matrix(value: Tensor, frames: int) -> Tensor:
    return F.interpolate(
        value.T[None], size=frames, mode="linear", align_corners=False
    )[0].T.contiguous()


def _resize_vector(value: Tensor, frames: int) -> Tensor:
    return F.interpolate(
        value.flatten()[None, None], size=frames, mode="linear", align_corners=False
    )[0, 0]


def _window_starts(frames: int, core: int, overlap: int) -> list[int]:
    if core <= overlap or frames <= 0:
        raise ValueError("invalid output window geometry")
    return list(range(0, frames, core - overlap))


def _core_weights(length: int, overlap: int, first: bool, last: bool) -> Tensor:
    weights = torch.ones(length)
    fade = min(overlap, length)
    if not first and fade:
        phase = torch.linspace(0, torch.pi / 2, fade)
        weights[:fade] = phase.sin().square()
    if not last and fade:
        phase = torch.linspace(0, torch.pi / 2, fade)
        weights[-fade:] = phase.cos().square()
    return weights


@torch.inference_mode()
def render_mel(
    system: HARPFlow,
    content: Tensor,
    f0: Tensor,
    rms: Tensor,
    noise: Tensor,
    speaker_id: int,
    speaker_code: Tensor | None,
    device: torch.device,
    *,
    visible: int,
    core: int,
    overlap: int,
    batch_size: int,
    steps: int,
    guidance_strength: float,
) -> Tensor:
    frames = content.shape[0]
    left_context = (visible - core) // 2
    starts = _window_starts(frames, core, overlap)
    accumulated = torch.zeros(frames, noise.shape[-1])
    weight_sum = torch.zeros(frames)
    for batch_start in range(0, len(starts), batch_size):
        selected = starts[batch_start : batch_start + batch_size]
        batch = {
            "content": torch.zeros(len(selected), visible, content.shape[-1]),
            "f0": torch.zeros(len(selected), visible, 1),
            "rms": torch.zeros(len(selected), visible, 1),
            "noise": torch.zeros(len(selected), visible, noise.shape[-1]),
            "mask": torch.zeros(len(selected), visible, dtype=torch.bool),
        }
        for index, start in enumerate(selected):
            visible_start = start - left_context
            source_start = max(0, visible_start)
            source_stop = min(frames, visible_start + visible)
            target_start = source_start - visible_start
            target_stop = target_start + source_stop - source_start
            batch["content"][index, target_start:target_stop] = content[
                source_start:source_stop
            ]
            batch["f0"][index, target_start:target_stop, 0] = f0[
                source_start:source_stop
            ]
            batch["rms"][index, target_start:target_stop, 0] = rms[
                source_start:source_stop
            ]
            batch["noise"][index, target_start:target_stop] = noise[
                source_start:source_stop
            ]
            batch["mask"][index, target_start:target_stop] = True
        batch = {name: value.to(device) for name, value in batch.items()}
        prediction = system.sample(
            batch["content"],
            batch["f0"],
            batch["rms"],
            torch.full((len(selected),), speaker_id, dtype=torch.long, device=device),
            batch["mask"],
            steps=steps,
            method="euler",
            guidance_strength=guidance_strength,
            initial_noise=batch["noise"],
            speaker_code_override=(
                None if speaker_code is None else speaker_code.expand(len(selected), -1)
            ),
        ).cpu()
        for index, start in enumerate(selected):
            stop = min(frames, start + core)
            length = stop - start
            window_start = left_context
            values = prediction[index, window_start : window_start + length]
            weights = _core_weights(
                length, overlap, first=start == 0, last=stop == frames
            )
            accumulated[start:stop] += values * weights[:, None]
            weight_sum[start:stop] += weights
    if bool((weight_sum <= 0).any()):
        raise RuntimeError("mel overlap-add left uncovered frames")
    return accumulated / weight_sum[:, None]


@torch.inference_mode()
def vocode_chunked(
    vocoder,
    mel: Tensor,
    f0: Tensor,
    device: torch.device,
    hop_length: int,
    *,
    core_frames: int = 1024,
    context_frames: int = 64,
    overlap_frames: int = 4,
) -> Tensor:
    """Vocode long tracks with global excitation and waveform overlap-add."""
    if core_frames <= overlap_frames or overlap_frames < 0:
        raise ValueError("core_frames must be greater than overlap_frames")
    frames = mel.shape[0]
    if frames <= 0:
        return torch.empty(0)
    harmonic_source = prepare_pc_nsf_harmonic_source(vocoder, f0, device)
    stride = core_frames - overlap_frames
    output = torch.zeros(frames * hop_length)
    weight_sum = torch.zeros_like(output)

    for start in range(0, frames, stride):
        stop = min(mel.shape[0], start + core_frames)
        visible_start = max(0, start - context_frames)
        visible_stop = min(mel.shape[0], stop + context_frames)
        waveform = synthesize_pc_nsf(
            vocoder,
            mel[visible_start:visible_stop],
            f0[visible_start:visible_stop],
            device,
            harmonic_source=harmonic_source[
                :, :, visible_start * vocoder.upp : visible_stop * vocoder.upp
            ],
        )
        keep_start = (start - visible_start) * hop_length
        keep_length = (stop - start) * hop_length
        piece = waveform[keep_start : keep_start + keep_length]
        weights = torch.ones(keep_length)
        fade = min(overlap_frames * hop_length, keep_length)
        if start > 0 and fade:
            phase = torch.linspace(0, torch.pi / 2, fade)
            weights[:fade] = phase.sin().square()
        if stop < frames and fade:
            phase = torch.linspace(0, torch.pi / 2, fade)
            weights[-fade:] = phase.cos().square()
        output[start * hop_length : stop * hop_length] += piece * weights
        weight_sum[start * hop_length : stop * hop_length] += weights
    if bool((weight_sum <= 0).any()):
        raise RuntimeError("vocoder overlap-add left uncovered samples")
    return output / weight_sum


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SingerConverter:
    """One target singer, one device, reusable model and feature extraction."""

    def __init__(
        self,
        *,
        config: Path,
        finetune: Path,
        content_model: Path,
        pc_nsf_checkout: Path,
        pc_nsf_lock: Path,
        vocoder_checkpoint: Path,
        device: str = "auto",
        state: str = "ema",
    ):
        self.config = HARPConfig.load(config)
        self.device = resolve_device(device)
        self.runtime = configure_cuda(
            self.device,
            sdpa_backend=self.config.training.sdpa_backend,
            allow_tf32=self.config.training.allow_tf32,
        )
        checkpoint = torch.load(
            finetune, map_location="cpu", weights_only=False, mmap=True
        )
        if checkpoint.get("checkpoint_type") not in {
            "singer_finetune_audit_v1",
            "singer_finetune_full_v1",
            "singer_finetune_inference_v1",
        }:
            raise ValueError("expected a singer finetune checkpoint")
        if checkpoint["run"].get("config_sha256") != self.config.digest():
            raise ValueError("checkpoint config differs from selected config")
        if state not in {"raw", "ema"}:
            raise ValueError("state must be raw or ema")
        weights_key, code_key = (
            ("ema", "target_code_ema") if state == "ema" else ("model", "target_code")
        )
        if weights_key not in checkpoint or code_key not in checkpoint:
            raise ValueError(f"checkpoint does not contain {state} state")
        transform = FlowTransform.load(self.config.flow.transform_path).to(self.device)
        feature = FeatureContract.load(self.config.feature.contract_path)
        validate_feature_contract(
            feature, self.config.model, self.config.harmonic, self.config.feature
        )
        model = HARPCore(
            self.config.model, self.config.harmonic, feature, self.config.num_speakers
        ).to(self.device)
        model.load_state_dict(checkpoint[weights_key], strict=True)
        model.eval()
        self.system = HARPFlow(
            model,
            transform,
            speaker_drop_probability=0,
            lambda_floor=self.config.flow.lambda_floor,
            q_floor=self.config.flow.q_floor,
        ).eval()
        self.code = torch.as_tensor(checkpoint[code_key]).float().to(self.device)[None]
        self.finetune_sha256 = _sha256(finetune)
        self.state = state
        self.features = FeatureExtractor(content_model, self.config, self.device)
        self.vocoder, contract = load_pc_nsf(
            pc_nsf_checkout,
            vocoder_checkpoint,
            pc_nsf_lock,
            self.config,
            self.device,
        )
        self.vocoder_revision = contract.revision

    @torch.inference_mode()
    def convert(
        self,
        input_path: Path,
        *,
        steps: int,
        guidance: float,
        seed: int,
        window_batch_size: int,
    ) -> tuple[Tensor, dict]:
        if (
            steps <= 0
            or not math.isfinite(guidance)
            or guidance < 0
            or window_batch_size <= 0
        ):
            raise ValueError(
                "steps and batch size must be positive; "
                "guidance must be finite and nonnegative"
            )
        started = time.perf_counter()
        waveform, content, f0, rms, samples = self.features.extract(input_path)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            content.shape[0], self.config.model.mel_channels, generator=generator
        )
        mel = render_mel(
            self.system,
            content,
            f0,
            rms,
            noise,
            0,
            self.code,
            self.device,
            visible=768,
            core=384,
            overlap=64,
            batch_size=window_batch_size,
            steps=steps,
            guidance_strength=guidance,
        )
        converted = vocode_chunked(
            self.vocoder, mel, f0, self.device, self.config.feature.hop_length
        )
        if converted.numel() < samples:
            raise RuntimeError(
                f"vocoder output is short: {converted.numel()}/{samples}"
            )
        converted = converted[:samples].float().cpu()
        if not torch.isfinite(converted).all():
            raise ValueError("conversion produced non-finite samples")
        return converted, {
            "artifact_type": "rift_harp_singer_conversion_v3",
            "input_sha256": _sha256(input_path),
            "finetune_sha256": self.finetune_sha256,
            "state": self.state,
            "sample_rate": self.config.feature.sample_rate,
            "source_samples": samples,
            "source_frames": content.shape[0],
            "source_peak": float(waveform.abs().max()),
            "peak": float(converted.abs().max()),
            "params": {
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                "window_batch_size": window_batch_size,
            },
            "vocoder_revision": self.vocoder_revision,
            "runtime": self.runtime,
            "elapsed_seconds": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--finetune", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--content-model", type=Path, required=True)
    parser.add_argument("--pc-nsf-checkout", type=Path, required=True)
    parser.add_argument("--pc-nsf-lock", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--state", choices=("raw", "ema"), default="ema")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    import soundfile as sf

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converter = SingerConverter(
        config=args.config,
        finetune=args.finetune,
        content_model=args.content_model,
        pc_nsf_checkout=args.pc_nsf_checkout,
        pc_nsf_lock=args.pc_nsf_lock,
        vocoder_checkpoint=args.vocoder_checkpoint,
        device=args.device,
        state=args.state,
    )
    audio, record = converter.convert(
        args.input,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        window_batch_size=args.window_batch_size,
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    )
    try:
        sf.write(
            staging / "converted.wav",
            audio.numpy(),
            record["sample_rate"],
            subtype="FLOAT",
        )
        record["output_sha256"] = _sha256(staging / "converted.wav")
        (staging / "manifest.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.renameat2(
            -100, os.fsencode(staging), -100, os.fsencode(args.output), 1
        ):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(args.output))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
