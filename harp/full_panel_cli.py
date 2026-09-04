from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from .checkpoint import validate_checkpoint_contract, validate_external_artifacts
from .config import HARPConfig
from .data import FeatureDataset, SampleRequest, collate_features
from .endpoint_audit_cli import _fixed_noise
from .feature_contract import FeatureContract
from .flow import HARPFlow
from .flow_transform import FlowTransform
from .manifest import load_manifest, manifest_sha256
from .model import HARPCore
from .panel_cli import validate_panel_features
from .performance import compile_model_in_place, configure_cuda
from .precision import configure_heavy_linears
from .vocoder import load_pc_nsf, synthesize_pc_nsf


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the HARP PC-NSF full panel")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pc-nsf-checkout", type=Path, required=True)
    parser.add_argument("--pc-nsf-lock", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--panel", default="song_disjoint_shadow")
    parser.add_argument("--samples-per-length", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--method", choices=("euler", "heun"), default="euler")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples_per_length <= 0 or args.steps <= 0:
        raise ValueError("sample and solver counts must be positive")

    config = HARPConfig.load(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_contract(checkpoint["contract"], config)
    validate_external_artifacts(
        checkpoint["contract"],
        transform_path=config.flow.transform_path,
        transform_audit_path=config.flow.audit_path,
        feature_contract_path=config.feature.contract_path,
        manifest_path=args.manifest,
    )
    panel_artifact = json.loads(args.panels.read_text(encoding="utf-8"))
    if panel_artifact.get("manifest_sha256") != manifest_sha256(args.manifest):
        raise ValueError("fixed panel manifest hash differs")
    panel_items = panel_artifact["panels"].get(args.panel)
    if not panel_items:
        raise ValueError(f"panel is empty or missing: {args.panel}")

    raw_manifest = _load_raw_manifest(args.manifest)
    entries = load_manifest(args.manifest)
    id_to_index = {entry.id: index for index, entry in enumerate(entries)}
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        speaker_to_id=checkpoint["speaker_to_id"],
        voiced_crop_probability=config.training.voiced_crop_probability,
    )
    selected = _select_by_length(panel_items, args.samples_per_length)
    batches = _load_panel_batches(dataset, id_to_index, selected)

    device = torch.device(args.device)
    if device.type == "cuda":
        configure_cuda(
            device,
            sdpa_backend=config.training.sdpa_backend,
            allow_tf32=config.training.allow_tf32,
        )
    transform = FlowTransform.load(config.flow.transform_path).to(device)
    feature_contract = FeatureContract.load(config.feature.contract_path)
    model = HARPCore(
        config.model, config.harmonic, feature_contract, config.num_speakers
    ).to(device)
    configure_heavy_linears(model, config.model.heavy_linear_precision)
    if device.type == "cuda":
        compile_model_in_place(model, "default")
    system = HARPFlow(
        model,
        transform,
        speaker_drop_probability=0,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    ).eval()
    vocoder, vocoder_contract = load_pc_nsf(
        args.pc_nsf_checkout,
        args.vocoder_checkpoint,
        args.pc_nsf_lock,
        config,
        device,
    )
    pitch_model = _load_pitch_model(device)

    if args.output.exists():
        raise FileExistsError(f"full-panel output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(dir=args.output.parent, prefix=".full-panel."))
    try:
        results = _render(
            destination,
            batches,
            raw_manifest,
            {"raw": checkpoint["model"], "ema": checkpoint["ema"]},
            model,
            system,
            vocoder,
            pitch_model,
            config,
            device,
            args.steps,
            args.method,
        )
        payload = {
            "artifact_type": "harp_full_panel_v1",
            "created_at_unix": time.time(),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "checkpoint_progress": checkpoint["progress"],
            "panel_artifact": str(args.panels),
            "panel_sha256": _sha256(args.panels),
            "manifest_sha256": manifest_sha256(args.manifest),
            "vocoder": {
                "checkout_revision": vocoder_contract.revision,
                "checkpoint_sha256": vocoder_contract.checkpoint_sha256,
            },
            "solver": {"method": args.method, "steps": args.steps},
            "results": results,
            "aggregate_pitch": _aggregate_pitch(results),
            "aggregate_tail": _aggregate_tail(results),
        }
        (destination / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(destination, args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
    except BaseException:
        for path in destination.glob("*"):
            path.unlink(missing_ok=True)
        destination.rmdir()
        raise


@torch.inference_mode()
def _render(
    destination: Path,
    batches: dict[int, tuple[list[dict], dict[str, Tensor]]],
    raw_manifest: dict[str, dict],
    states: dict[str, dict[str, Tensor]],
    model: HARPCore,
    system: HARPFlow,
    vocoder,
    pitch_model,
    config: HARPConfig,
    device: torch.device,
    steps: int,
    method: str,
) -> list[dict]:
    import soundfile as sf

    results = []
    reference_rates: dict[str, int] = {}
    for state_name in ("raw", "ema"):
        model.load_state_dict(states[state_name], strict=True)
        model.eval()
        for length, (items, cpu_batch) in sorted(batches.items()):
            batch = {name: value.to(device) for name, value in cpu_batch.items()}
            prediction = system.sample(
                batch["content"],
                batch["f0"],
                batch["rms"],
                batch["speaker"],
                batch["mask"],
                steps=steps,
                method=method,
                guidance_strength=1.0,
                initial_noise=_fixed_noise(batch, config.model.mel_channels, device),
            )
            for index, item in enumerate(items):
                frames = int(batch["length"][index])
                target_f0 = batch["f0"][index, :frames, 0].float().cpu()
                waveform = synthesize_pc_nsf(
                    vocoder, prediction[index, :frames], target_f0, device
                )
                generated_f0 = pitch_model.infer(
                    waveform.to(device)[None, :, None],
                    sr=config.feature.sample_rate,
                    decoder_mode="local_argmax",
                    threshold=0.006,
                    f0_min=config.harmonic.f0_min,
                    f0_max=config.harmonic.f0_max,
                    interp_uv=False,
                    output_interp_target_length=frames,
                )
                generated_f0 = torch.as_tensor(generated_f0).squeeze().float().cpu()
                stem = f"{length}-{item['entry_id']}-{int(item['crop_start'])}"
                reference_name = f"{stem}-reference.wav"
                generated_name = f"{stem}-{state_name}.wav"
                if reference_name not in reference_rates:
                    source = raw_manifest[item["entry_id"]]
                    source_path = Path(source["audio_path"])
                    if _sha256(source_path) != source["audio_sha256"]:
                        raise ValueError(f"source audio changed: {item['entry_id']}")
                    reference, source_sample_rate = _reference_crop(
                        source_path,
                        int(item["crop_start"]),
                        frames,
                        config.feature.sample_rate,
                        config.feature.hop_length,
                    )
                    sf.write(
                        destination / reference_name,
                        reference.numpy(),
                        config.feature.sample_rate,
                        subtype="PCM_24",
                    )
                    reference_rates[reference_name] = source_sample_rate
                sf.write(
                    destination / generated_name,
                    waveform.numpy(),
                    config.feature.sample_rate,
                    subtype="PCM_24",
                )
                results.append(
                    {
                        "state": state_name,
                        "entry_id": item["entry_id"],
                        "dataset": item["dataset"],
                        "speaker": item["speaker"],
                        "song": item["song"],
                        "requested_frames": length,
                        "actual_frames": frames,
                        "crop_start": int(item["crop_start"]),
                        "noise_seed": int(item["noise_seed"]),
                        "generated": generated_name,
                        "reference": reference_name,
                        "reference_source_sample_rate": reference_rates[
                            reference_name
                        ],
                        "pitch": pitch_metrics(target_f0, generated_f0),
                        "tail": tail_metrics(
                            waveform,
                            frames * config.feature.hop_length,
                            config.feature.hop_length,
                        ),
                    }
                )
    return results


def _reference_crop(
    path: Path,
    start_frame: int,
    frames: int,
    target_sample_rate: int,
    hop_length: int,
) -> tuple[Tensor, int]:
    import soundfile as sf
    import torchaudio.functional as audio_functional

    audio, source_sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"panel reference must be mono: {path}")
    waveform = torch.from_numpy(audio[:, 0])
    if source_sample_rate != target_sample_rate:
        waveform = audio_functional.resample(
            waveform, source_sample_rate, target_sample_rate
        )
    start_sample = start_frame * hop_length
    wanted = frames * hop_length
    crop = waveform[start_sample : start_sample + wanted]
    return torch.nn.functional.pad(crop, (0, max(0, wanted - crop.numel()))), int(
        source_sample_rate
    )


def _load_panel_batches(
    dataset: FeatureDataset, id_to_index: dict[str, int], items: list[dict]
) -> dict[int, tuple[list[dict], dict[str, Tensor]]]:
    grouped: dict[int, list[tuple[dict, dict[str, Tensor]]]] = defaultdict(list)
    for item in items:
        entry = dataset.entries[id_to_index[item["entry_id"]]]
        validate_panel_features(entry, item)
        sample = dataset[
            SampleRequest(
                id_to_index[item["entry_id"]],
                int(item["requested_frames"]),
                int(item["noise_seed"]),
                int(item["crop_start"]),
            )
        ]
        sample["noise_seed"] = torch.tensor(int(item["noise_seed"]))
        grouped[int(item["requested_frames"])].append((item, sample))
    return {
        length: ([item for item, _ in values], collate_features([x for _, x in values]))
        for length, values in grouped.items()
    }


def _select_by_length(items: list[dict], samples_per_length: int) -> list[dict]:
    selected = []
    counts: dict[int, int] = defaultdict(int)
    for item in items:
        length = int(item["requested_frames"])
        if counts[length] < samples_per_length:
            selected.append(item)
            counts[length] += 1
    return selected


def _load_raw_manifest(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                result[str(payload["id"])] = payload
    return result


def _load_pitch_model(device: torch.device):
    try:
        from torchfcpe import spawn_bundled_infer_model
    except ImportError as error:
        raise RuntimeError("full-panel evaluation requires torchfcpe") from error
    return spawn_bundled_infer_model(device=str(device))


def pitch_metrics(target_f0: Tensor, generated_f0: Tensor) -> dict[str, float | int]:
    target = target_f0.float().flatten()
    generated = generated_f0.float().flatten()
    frames = min(target.numel(), generated.numel())
    target, generated = target[:frames], generated[:frames]
    target_voiced, generated_voiced = target > 0, generated > 0
    both = target_voiced & generated_voiced
    true_positive = int(both.sum())
    precision = true_positive / max(1, int(generated_voiced.sum()))
    recall = true_positive / max(1, int(target_voiced.sum()))
    cents = 1200 * torch.log2(generated[both] / target[both]) if both.any() else None
    return {
        "frames": frames,
        "voicing_precision": precision,
        "voicing_recall": recall,
        "voicing_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "f0_cents_mae": float(cents.abs().mean()) if cents is not None else 0.0,
        "gross_pitch_error_ratio": (
            float((cents.abs() > 50).float().mean()) if cents is not None else 0.0
        ),
    }


def tail_metrics(waveform: Tensor, expected_samples: int, hop_length: int) -> dict:
    waveform = waveform.float().flatten()
    tail = waveform[-min(hop_length, waveform.numel()) :]
    return {
        "samples": waveform.numel(),
        "expected_samples": expected_samples,
        "length_error": waveform.numel() - expected_samples,
        "finite": bool(torch.isfinite(waveform).all()),
        "peak": float(waveform.abs().max()) if waveform.numel() else 0.0,
        "tail_rms": float(tail.square().mean().sqrt()) if tail.numel() else 0.0,
        "tail_peak": float(tail.abs().max()) if tail.numel() else 0.0,
    }


def _aggregate_pitch(results: list[dict]) -> dict[str, float]:
    names = (
        "voicing_precision",
        "voicing_recall",
        "voicing_f1",
        "f0_cents_mae",
        "gross_pitch_error_ratio",
    )
    return {
        f"{state}/{name}": sum(
            item["pitch"][name] for item in results if item["state"] == state
        )
        / max(1, sum(item["state"] == state for item in results))
        for state in ("raw", "ema")
        for name in names
    }


def _aggregate_tail(results: list[dict]) -> dict[str, float]:
    return {
        f"{state}/maximum_tail_peak": max(
            item["tail"]["tail_peak"] for item in results if item["state"] == state
        )
        for state in ("raw", "ema")
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
