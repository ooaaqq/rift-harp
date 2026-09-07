"""Build offline pseudo-speaker ContentVec variants from a frozen HARP teacher."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import validate_checkpoint_contract
from .config import HARPConfig
from .data import _matrix, _vector
from .feature_contract import FeatureContract, validate_feature_contract
from .flow import HARPFlow
from .flow_transform import FlowTransform
from .manifest import load_manifest, manifest_sha256
from .model import HARPCore
from .performance import configure_cuda, resolve_device
from .singer_conversion import (
    FrozenContentEncoder,
    _resize_matrix,
    encode_content,
    render_mel,
    vocode_chunked,
)
from .vocoder import load_pc_nsf


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base: int, recording_id: str) -> int:
    value = hashlib.sha256(f"{base}:{recording_id}".encode()).digest()
    return int.from_bytes(value[:8], "little") % (2**63)


def _tree_sha256(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_target_features(entry, config: HARPConfig) -> tuple[torch.Tensor, ...]:
    prefix = Path(entry.feature_prefix)
    content_path = entry.content_feature_path or f"{prefix}.content.pt"
    content = _matrix(
        torch.load(content_path, map_location="cpu", weights_only=True),
        config.model.content_dim,
    ).float()
    mel = _matrix(
        torch.load(f"{prefix}.mel.pt", map_location="cpu", weights_only=True),
        config.model.mel_channels,
    ).float()
    f0 = _vector(
        torch.load(f"{prefix}.f0.pt", map_location="cpu", weights_only=True)
    ).float()
    rms = _vector(
        torch.load(f"{prefix}.rms.pt", map_location="cpu", weights_only=True)
    ).float()
    content = _resize_matrix(content, entry.frames)
    f0 = f0.flatten()[: entry.frames]
    rms = rms.flatten()[: entry.frames]
    if (
        mel.shape[0] != entry.frames
        or f0.numel() != entry.frames
        or rms.numel() != entry.frames
    ):
        raise ValueError(f"{entry.id}: target feature lengths differ")
    if content.shape != (entry.frames, config.model.content_dim):
        raise ValueError(f"{entry.id}: target content shape differs")
    return content, f0, rms, mel


def build_bank(config: HARPConfig, args: argparse.Namespace) -> None:
    import soundfile as sf
    import torchaudio.functional as AF

    output = Path(args.output).resolve()
    bank_path = output / "bank.json"
    if output.exists() and not args.resume:
        raise FileExistsError("pseudo-bank output directory already exists")
    if args.resume and not bank_path.is_file():
        raise FileNotFoundError("resume requires an existing pseudo bank")
    (output / "waveforms").mkdir(parents=True, exist_ok=args.resume)
    (output / "content").mkdir(exist_ok=args.resume)
    device = resolve_device(args.device)
    configure_cuda(
        device,
        sdpa_backend=config.training.sdpa_backend,
        allow_tf32=config.training.allow_tf32,
    )
    parent_path = Path(args.parent)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False, mmap=True)
    validate_checkpoint_contract(parent["contract"], config)
    feature = FeatureContract.load(config.feature.contract_path)
    validate_feature_contract(feature, config.model, config.harmonic, config.feature)
    model = HARPCore(config.model, config.harmonic, feature, config.num_speakers)
    model.load_state_dict(parent["ema"], strict=True)
    model.to(device).eval()
    system = HARPFlow(
        model,
        FlowTransform.load(config.flow.transform_path).to(device),
        speaker_drop_probability=0.0,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    ).eval()
    speakers = parent["speaker_to_id"]
    missing = sorted(set(args.carrier_speaker) - speakers.keys())
    if missing:
        raise ValueError(f"unknown carrier speakers: {missing}")
    vocoder, vocoder_contract = load_pc_nsf(
        args.pc_nsf_checkout,
        args.vocoder_checkpoint,
        args.pc_nsf_lock,
        config,
        device,
    )
    encoder = FrozenContentEncoder.load(args.content_model, config.model.content_dim)
    encoder.to(device).eval()
    entries = [
        entry
        for entry in load_manifest(args.manifest)
        if entry.quality_status == "accepted"
        and entry.split in set(args.splits)
        and (not args.recording_id or entry.id in set(args.recording_id))
    ]
    expected_contract = {
        "artifact_type": "rift_harp_pseudo_content_bank_v1",
        "teacher_checkpoint_sha256": _sha256(parent_path),
        "target_manifest_sha256": manifest_sha256(args.manifest),
        "student_target_singer_key": args.student_target,
        "carriers": [
            {"speaker_key": name, "speaker_id": int(speakers[name])}
            for name in args.carrier_speaker
        ],
        "generation": {
            "solver": "euler",
            "steps": args.steps,
            "guidance": 1.0,
            "key_shift": 0,
            "seed": args.seed,
            "window": {"visible": 768, "core": 384, "overlap": 64},
        },
        "content_model_sha256": _tree_sha256(args.content_model),
        "vocoder_checkpoint_sha256": _sha256(args.vocoder_checkpoint),
        "feature_contract_sha256": _sha256(config.feature.contract_path),
        "flow_transform_sha256": _sha256(config.flow.transform_path),
    }
    if args.resume:
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        for key, value in expected_contract.items():
            if bank.get(key) != value:
                raise ValueError(f"resume pseudo-bank contract differs: {key}")
    else:
        bank = {
            **expected_contract,
            "teacher_checkpoint": str(parent_path),
            "teacher_state": "ema",
            "target_manifest": str(args.manifest),
            "content_model": str(args.content_model),
            "vocoder_checkpoint": str(args.vocoder_checkpoint),
            "vocoder_revision": vocoder_contract.revision,
            "quality_policy": "accepted_by_flag"
            if args.accept_generated
            else "pending_manual_review",
            "variants": [],
        }
        _atomic_json(bank_path, bank)
    completed = {
        (item["origin_target_recording_id"], item["carrier_speaker_key"])
        for item in bank["variants"]
    }
    for entry in entries:
        content, f0, rms, target_mel = _load_target_features(entry, config)
        generator = torch.Generator(device="cpu").manual_seed(
            _stable_seed(args.seed, entry.id)
        )
        noise = torch.randn(
            entry.frames, config.model.mel_channels, generator=generator
        )
        target_feature_sha = hashlib.sha256(
            b"".join(
                tensor.contiguous().numpy().tobytes()
                for tensor in (content, f0, rms, target_mel)
            )
        ).hexdigest()
        for carrier in args.carrier_speaker:
            if (entry.id, carrier) in completed:
                continue
            token = carrier.replace(":", "-").replace("/", "-")
            waveform_path = output / "waveforms" / f"{entry.id}--{token}.wav"
            content_path = output / "content" / f"{entry.id}--{token}.content.pt"
            mel = render_mel(
                system,
                content,
                f0,
                rms,
                noise,
                int(speakers[carrier]),
                None,
                device,
                visible=768,
                core=384,
                overlap=64,
                batch_size=args.window_batch_size,
                steps=args.steps,
                guidance_strength=1.0,
            )
            waveform = vocode_chunked(
                vocoder, mel, f0, device, config.feature.hop_length
            )
            expected_samples = entry.frames * config.feature.hop_length
            waveform = F.pad(
                waveform, (0, max(0, expected_samples - waveform.numel()))
            )[:expected_samples]
            sf.write(
                waveform_path,
                waveform.numpy(),
                config.feature.sample_rate,
                subtype="PCM_24",
            )
            content_waveform = AF.resample(waveform, config.feature.sample_rate, 16_000)
            if device.type == "cuda":
                torch.backends.cuda.enable_cudnn_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            pseudo_content = (
                encode_content(encoder, content_waveform, 16_000, device).float().cpu()
            )
            configure_cuda(
                device,
                sdpa_backend=config.training.sdpa_backend,
                allow_tf32=config.training.allow_tf32,
            )
            pseudo_content = _resize_matrix(pseudo_content, entry.frames)
            torch.save(pseudo_content, content_path)
            item = {
                "origin_target_recording_id": entry.id,
                "origin_target_song": entry.song,
                "origin_target_split": entry.split,
                "student_target_singer_key": args.student_target,
                "carrier_speaker_key": carrier,
                "carrier_speaker_id": int(speakers[carrier]),
                "target_frames": entry.frames,
                "target_feature_sha256": target_feature_sha,
                "waveform_path": str(waveform_path),
                "waveform_sha256": _sha256(waveform_path),
                "content_feature_path": str(content_path),
                "content_sha256": _sha256(content_path),
                "quality_status": "accepted" if args.accept_generated else "pending",
                "peak": float(waveform.abs().max()),
            }
            bank["variants"].append(item)
            _atomic_json(bank_path, bank)
            print(json.dumps(item), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-target", default="custom:target")
    parser.add_argument("--carrier-speaker", action="append", required=True)
    parser.add_argument("--recording-id", action="append")
    parser.add_argument("--splits", nargs="+", default=("train", "validation"))
    parser.add_argument("--content-model", type=Path, required=True)
    parser.add_argument("--pc-nsf-checkout", type=Path, required=True)
    parser.add_argument("--pc-nsf-lock", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--accept-generated", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    build_bank(HARPConfig.load(args.config), args)


if __name__ == "__main__":
    main()
