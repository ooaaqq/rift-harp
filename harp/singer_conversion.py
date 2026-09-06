"""Offline conversion with a foundation speaker or target finetune."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .checkpoint import validate_checkpoint_contract
from .config import HARPConfig
from .feature_contract import FeatureContract, validate_feature_contract
from .flow import HARPFlow
from .flow_transform import FlowTransform
from .model import HARPCore
from .performance import configure_cuda
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


@torch.inference_mode()
def extract_features(
    input_path: Path,
    content_model: Path,
    config: HARPConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    import soundfile as sf
    import torchaudio.functional as AF
    from torchfcpe import spawn_bundled_infer_model

    audio, source_rate = sf.read(input_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio).mean(dim=1)
    if source_rate != config.feature.sample_rate:
        waveform = AF.resample(waveform, source_rate, config.feature.sample_rate)
    expected_samples = waveform.numel()
    frames = expected_samples // config.feature.hop_length
    if frames <= 0:
        raise ValueError("input is shorter than one HARP frame")

    content_waveform = AF.resample(waveform, config.feature.sample_rate, 16_000)
    encoder = FrozenContentEncoder.load(content_model, config.model.content_dim)
    encoder.to(device).eval()
    content = encode_content(encoder, content_waveform, 16_000, device)
    del encoder
    content = _resize_matrix(content, frames)

    pitch_model = spawn_bundled_infer_model(device=str(device))
    f0 = pitch_model.infer(
        waveform.clamp(-1, 1).to(device)[None, :, None],
        sr=config.feature.sample_rate,
        decoder_mode="local_argmax",
        threshold=0.006,
        f0_min=config.harmonic.f0_min,
        f0_max=config.harmonic.f0_max,
        interp_uv=False,
        output_interp_target_length=frames,
    )
    f0 = _resize_vector(torch.as_tensor(f0).squeeze().float().cpu(), frames)
    del pitch_model

    window = config.feature.win_length
    pad = (window - config.feature.hop_length) // 2
    padded = F.pad(waveform[None, None], (pad, pad), mode="reflect")[0, 0]
    rms = padded.unfold(0, window, config.feature.hop_length).square().mean(-1).sqrt()
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


def _resolve_foundation_speakers(
    requested: list[str], speaker_to_id: dict[str, int]
) -> list[tuple[str, int]]:
    missing = [speaker for speaker in requested if speaker not in speaker_to_id]
    if missing:
        available = ", ".join(sorted(speaker_to_id))
        raise ValueError(
            f"unknown foundation speaker(s): {', '.join(missing)}; "
            f"available speakers: {available}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("foundation speakers must be unique")
    return [(speaker, int(speaker_to_id[speaker])) for speaker in requested]


def _filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return component or "speaker"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--parent", type=Path)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--finetune", type=Path, action="append")
    target.add_argument("--foundation-speaker", action="append")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--content-model", type=Path, required=True)
    parser.add_argument("--pc-nsf-checkout", type=Path, required=True)
    parser.add_argument("--pc-nsf-lock", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--states", nargs="+", choices=("raw", "ema"), default=("raw", "ema")
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, nargs="+", default=(1.0,))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(value <= 0 for value in args.guidance):
        parser.error("--guidance values must be positive")
    if args.foundation_speaker and args.parent is None:
        parser.error("--parent is required with --foundation-speaker")

    import soundfile as sf

    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    args.output.mkdir(parents=True)
    config = HARPConfig.load(args.config)
    device = torch.device(args.device)
    parent = (
        torch.load(args.parent, map_location="cpu", weights_only=False, mmap=True)
        if args.parent is not None
        else None
    )
    if parent is not None:
        validate_checkpoint_contract(parent["contract"], config)
    transform = FlowTransform.load(config.flow.transform_path).to(device)
    feature = FeatureContract.load(config.feature.contract_path)
    validate_feature_contract(feature, config.model, config.harmonic, config.feature)
    model = HARPCore(config.model, config.harmonic, feature, config.num_speakers).to(
        device
    )
    if parent is not None:
        model.load_state_dict(parent["ema"], strict=True)
    model.eval()
    system = HARPFlow(
        model,
        transform,
        speaker_drop_probability=0,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    ).eval()
    waveform, content, f0, rms, expected_samples = extract_features(
        args.input, args.content_model, config, device
    )
    runtime = configure_cuda(
        device,
        sdpa_backend=config.training.sdpa_backend,
        allow_tf32=config.training.allow_tf32,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(
        content.shape[0], config.model.mel_channels, generator=generator
    )
    vocoder, vocoder_contract = load_pc_nsf(
        args.pc_nsf_checkout,
        args.vocoder_checkpoint,
        args.pc_nsf_lock,
        config,
        device,
    )
    results = []
    started = time.time()
    if args.foundation_speaker:
        assert parent is not None
        targets = [
            {
                "mode": "foundation",
                "name": name,
                "speaker_id": speaker_id,
                "state": "ema",
                "code": None,
                "model_state": parent["ema"],
            }
            for name, speaker_id in _resolve_foundation_speakers(
                args.foundation_speaker, parent["speaker_to_id"]
            )
        ]
    else:
        targets = []
        for finetune_path in args.finetune:
            finetune = torch.load(
                finetune_path, map_location="cpu", weights_only=False, mmap=True
            )
            checkpoint_type = finetune.get("checkpoint_type")
            if checkpoint_type not in {
                "singer_finetune_audit_v1",
                "singer_finetune_full_v1",
                "singer_finetune_inference_v1",
            }:
                raise ValueError("--finetune must be a singer finetune checkpoint")
            if finetune["run"].get("config_sha256") != config.digest():
                raise ValueError("finetune checkpoint config does not match --config")
            if parent is not None:
                parent_sha256 = parent.get("source_checkpoint_sha256") or _sha256(
                    args.parent
                )
                if finetune["run"].get("parent_checkpoint_sha256") != parent_sha256:
                    raise ValueError("finetune checkpoint belongs to another parent")
            states = (
                ("ema",)
                if checkpoint_type == "singer_finetune_inference_v1"
                else args.states
            )
            if "ema" not in finetune or "target_code_ema" not in finetune:
                raise ValueError(
                    "inference checkpoint must contain EMA model and target code"
                )
            for state in states:
                targets.append(
                    {
                        "mode": "finetune",
                        "name": finetune_path.stem,
                        "speaker_id": 0,
                        "state": state,
                        "code": torch.as_tensor(
                            finetune[
                                "target_code" if state == "raw" else "target_code_ema"
                            ]
                        )
                        .float()
                        .to(device)[None],
                        "model_state": finetune["model" if state == "raw" else "ema"],
                        "finetune": finetune,
                        "finetune_path": finetune_path,
                    }
                )
    for target_spec in targets:
        model.load_state_dict(target_spec["model_state"], strict=True)
        for guidance in args.guidance:
            mel = render_mel(
                system,
                content,
                f0,
                rms,
                noise,
                target_spec["speaker_id"],
                target_spec["code"],
                device,
                visible=768,
                core=384,
                overlap=64,
                batch_size=args.window_batch_size,
                steps=args.steps,
                guidance_strength=guidance,
            )
            converted = vocode_chunked(
                vocoder, mel, f0, device, config.feature.hop_length
            )
            converted = F.pad(
                converted, (0, max(0, expected_samples - converted.numel()))
            )
            converted = converted[:expected_samples]
            guidance_name = str(guidance).replace(".", "p")
            if target_spec["mode"] == "foundation":
                filename = (
                    f"foundation-{_filename_component(target_spec['name'])}"
                    f"-guidance-{guidance_name}.wav"
                )
                result = {
                    "target_mode": "foundation",
                    "foundation_speaker": target_spec["name"],
                    "speaker_id": target_spec["speaker_id"],
                    "foundation_state": "ema",
                }
            else:
                finetune = target_spec["finetune"]
                finetune_path = target_spec["finetune_path"]
                frames = int(finetune["progress"]["seen_target_valid_frames"])
                filename = (
                    f"finetune-{frames / 1_000_000:.3f}M-{target_spec['state']}"
                    f"-guidance-{guidance_name}.wav"
                )
                result = {
                    "target_mode": "finetune",
                    "finetune": str(finetune_path),
                    "finetune_sha256": _sha256(finetune_path),
                    "state": target_spec["state"],
                    "step": int(finetune["progress"]["global_step"]),
                    "seen_target_valid_frames": frames,
                }
            sf.write(
                args.output / filename,
                converted.numpy(),
                config.feature.sample_rate,
                subtype="PCM_24",
            )
            result.update(
                {
                    "guidance_strength": guidance,
                    "output": filename,
                    "output_sha256": _sha256(args.output / filename),
                    "peak": float(converted.abs().max()),
                }
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    manifest = {
        "artifact_type": "rift_harp_singer_conversion_v2",
        "target_mode": "foundation" if args.foundation_speaker else "finetune",
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "parent": str(args.parent) if args.parent is not None else None,
        "parent_sha256": _sha256(args.parent) if args.parent is not None else None,
        "solver": {"method": "euler", "steps": args.steps},
        "guidance_strengths": args.guidance,
        "seed": args.seed,
        "window": {"visible": 768, "core": 384, "overlap": 64},
        "source_frames": content.shape[0],
        "source_samples": expected_samples,
        "source_peak": float(waveform.abs().max()),
        "vocoder_revision": vocoder_contract.revision,
        "runtime": runtime,
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
