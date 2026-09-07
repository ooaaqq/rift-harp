import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from .checkpoint import validate_checkpoint_contract, validate_external_artifacts
from .config import HARPConfig
from .data import FeatureDataset, SampleRequest, collate_features
from .feature_contract import FeatureContract
from .flow import HARPFlow
from .flow_transform import FlowTransform, orthonormal_dct
from .manifest import load_manifest, manifest_sha256
from .model import HARPCore
from .panel_cli import validate_panel_features
from .performance import configure_cuda, resolve_device
from .vocoder import load_pc_nsf, synthesize_pc_nsf

LENGTHS = (256, 512, 768)
BANDS = {"dct_0_15": (0, 16), "dct_16_31": (16, 32), "dct_32_127": (32, 128)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HARP against frozen V3-null on Shadow-128"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--v3-baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--method", choices=("euler", "heun"), default="euler")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention-scale", type=float)
    parser.add_argument(
        "--model-states", nargs="+", choices=("raw", "ema"), default=("raw", "ema")
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("null", "correct"),
        default=("null", "correct"),
    )
    parser.add_argument("--skip-vocoder", action="store_true")
    parser.add_argument("--pc-nsf-checkout", type=Path)
    parser.add_argument("--pc-nsf-lock", type=Path)
    parser.add_argument("--vocoder-checkpoint", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.steps <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("batch, solver, and bootstrap counts must be positive")
    if args.output.exists():
        raise FileExistsError(f"comparison output already exists: {args.output}")
    vocoder_paths = (
        args.pc_nsf_checkout,
        args.pc_nsf_lock,
        args.vocoder_checkpoint,
    )
    if not args.skip_vocoder and any(path is None for path in vocoder_paths):
        raise ValueError("PC-NSF paths are required unless --skip-vocoder is set")

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
    panel = json.loads(args.panel_lock.read_text(encoding="utf-8"))
    baseline = json.loads(args.v3_baseline.read_text(encoding="utf-8"))
    _validate_protocol(
        panel,
        baseline,
        args.panel_lock,
        args.manifest,
        checkpoint["speaker_to_id"],
    )
    if args.steps != int(baseline["protocol"]["intervals"]):
        raise ValueError("solver intervals must match the frozen V3 baseline")
    if args.method != "euler":
        raise ValueError("solver method must match the frozen V3 Euler baseline")

    entries = load_manifest(args.manifest)
    id_to_index = {entry.id: index for index, entry in enumerate(entries)}
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        speaker_to_id=checkpoint["speaker_to_id"],
        voiced_crop_probability=config.training.voiced_crop_probability,
    )
    samples = sorted(panel["samples"], key=lambda item: int(item["ordinal"]))
    _validate_samples(samples, dataset, id_to_index)
    noise_seed = int(baseline["protocol"]["noise_seed"])
    noise = fixed_panel_noise(
        len(samples), max(LENGTHS), config.model.mel_channels, noise_seed
    )

    device = resolve_device(args.device)
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
    if args.attention_scale is not None:
        if not math.isfinite(args.attention_scale) or args.attention_scale <= 0:
            raise ValueError("attention scale must be finite and positive")
        for block in model.blocks:
            block.attention.scale = args.attention_scale
    system = HARPFlow(
        model,
        transform,
        speaker_drop_probability=0,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    ).eval()
    dct = orthonormal_dct(config.model.mel_channels).float()
    vocoder = None
    vocoder_contract = None
    if not args.skip_vocoder:
        vocoder, vocoder_contract = load_pc_nsf(
            args.pc_nsf_checkout,
            args.vocoder_checkpoint,
            args.pc_nsf_lock,
            config,
            device,
        )

    models = {"v3_null": _load_v3_baseline(baseline, samples)}
    checkpoint_states = {"raw": checkpoint["model"], "ema": checkpoint["ema"]}
    for state_name in args.model_states:
        state = checkpoint_states[state_name]
        model.load_state_dict(state, strict=True)
        model.eval()
        for condition in args.conditions:
            name = f"harp_{state_name}_{condition}"
            models[name] = _evaluate_harp(
                samples,
                dataset,
                id_to_index,
                noise,
                model,
                system,
                dct,
                vocoder,
                config,
                device,
                condition,
                args.batch_size,
                args.steps,
                args.method,
            )

    comparisons = {}
    for model_name in models:
        if model_name == "v3_null":
            continue
        comparisons[f"{model_name}_minus_v3_null"] = {
            str(length): paired_comparison(
                models["v3_null"][str(length)]["samples"],
                models[model_name][str(length)]["samples"],
                samples,
                bootstrap_samples=args.bootstrap_samples,
                seed=noise_seed + frame_index * 100 + 1,
            )
            for frame_index, length in enumerate(LENGTHS)
        }

    payload = {
        "artifact_type": "harp_shadow_128_comparison_v1",
        "created_at_unix": time.time(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_progress": checkpoint["progress"],
        "panel_lock": str(args.panel_lock),
        "panel_lock_sha256": _sha256(args.panel_lock),
        "v3_baseline": str(args.v3_baseline),
        "v3_baseline_sha256": _sha256(args.v3_baseline),
        "manifest_sha256": manifest_sha256(args.manifest),
        "protocol": {
            "samples": len(samples),
            "lengths": list(LENGTHS),
            "solver": args.method,
            "intervals": args.steps,
            "noise_seed": noise_seed,
            "noise_shape": list(noise.shape),
            "noise_generation": "torch CPU Generator float32; length prefixes",
            "active_mask": "valid and RMS > 0.001",
            "primary_metric": "per-sample active raw-log-mel MSE",
            "bootstrap": "dataset-song grouped",
            "bootstrap_samples": args.bootstrap_samples,
            "attention_scale": (
                1.0 / math.sqrt(config.model.head_dim)
                if args.attention_scale is None
                else args.attention_scale
            ),
            "model_states": list(args.model_states),
            "conditions": list(args.conditions),
            "catastrophe_threshold_full_raw_mse": float(
                baseline["protocol"]["catastrophe_threshold_raw_mse"]
            ),
        },
        "vocoder": (
            None
            if vocoder_contract is None
            else {
                "checkout_revision": vocoder_contract.revision,
                "checkpoint_sha256": vocoder_contract.checkpoint_sha256,
            }
        ),
        "models": models,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, payload)
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))


def fixed_panel_noise(samples: int, frames: int, channels: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        samples, frames, channels, generator=generator, dtype=torch.float32
    )


def sample_metrics(
    prediction: Tensor,
    target: Tensor,
    rms: Tensor,
    valid: Tensor,
    dct: Tensor,
    *,
    catastrophe_threshold: float = 5.0,
) -> dict[str, object]:
    prediction = prediction.float().cpu()
    target = target.float().cpu()
    rms = rms.float().cpu().flatten()
    valid = valid.bool().cpu().flatten()
    active = valid & (rms > 0.001)
    silence = valid & ~active
    error = (prediction - target).square()
    prediction_dct = prediction @ dct.T
    target_dct = target @ dct.T
    result: dict[str, object] = {
        "valid_frames": int(valid.sum()),
        "active_frames": int(active.sum()),
        "silence_frames": int(silence.sum()),
        "active_raw_mse": _selected_mse(error, active),
        "full_raw_mse": _selected_mse(error, valid),
        "silence_raw_mse": _selected_mse(error, silence),
    }
    for name, (start, end) in BANDS.items():
        result[f"{name}_mse"] = _selected_mse(
            (prediction_dct[:, start:end] - target_dct[:, start:end]).square(),
            active,
        )
    pred_high = prediction_dct[:, 32:][active].flatten()
    target_high = target_dct[:, 32:][active].flatten()
    result["dct_32_127_corr"] = _correlation(pred_high, target_high)
    result["dct_32_127_rms_ratio"] = _rms_ratio(pred_high, target_high)
    full_mse = result["full_raw_mse"]
    result["catastrophe"] = full_mse is not None and full_mse > catastrophe_threshold
    return result


def waveform_dbfs(waveform: Tensor) -> dict[str, float | bool | None]:
    waveform = waveform.float().flatten()
    finite = bool(torch.isfinite(waveform).all())
    if not finite or waveform.numel() == 0:
        return {"finite": finite, "rms_dbfs": None, "peak_dbfs": None}
    tiny = torch.finfo(torch.float32).tiny
    rms = waveform.square().mean().sqrt().clamp_min(tiny)
    peak = waveform.abs().max().clamp_min(tiny)
    return {
        "finite": True,
        "rms_dbfs": float(20 * torch.log10(rms)),
        "peak_dbfs": float(20 * torch.log10(peak)),
    }


def paired_comparison(
    before: list[dict],
    after: list[dict],
    panel_samples: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    panel_by_key = {
        (int(item["ordinal"]), str(item["entry_id"])): item for item in panel_samples
    }
    before_by_key = {_sample_key(item): item for item in before}
    after_by_key = {_sample_key(item): item for item in after}
    if before_by_key.keys() != after_by_key.keys():
        raise ValueError("paired model outputs contain different samples")
    grouped: dict[str, list[float]] = defaultdict(list)
    deltas = []
    wins = 0
    for key in sorted(before_by_key):
        before_value = before_by_key[key].get("active_raw_mse")
        after_value = after_by_key[key].get("active_raw_mse")
        if before_value is None or after_value is None:
            continue
        delta = float(after_value) - float(before_value)
        deltas.append(delta)
        wins += after_value < before_value
        grouped[str(panel_by_key[key]["song_key"])].append(delta)
    if not deltas:
        raise ValueError("paired comparison has no active samples")
    return {
        "definition": "HARP minus V3-null; lower is better",
        "samples": len(deltas),
        "song_units": len(grouped),
        "mean_active_mse_gap": statistics.fmean(deltas),
        "bootstrap_95_ci": grouped_bootstrap_ci(
            grouped, bootstrap_samples=bootstrap_samples, seed=seed
        ),
        "harp_win_rate": wins / len(deltas),
    }


def grouped_bootstrap_ci(
    grouped: dict[str, list[float]], *, bootstrap_samples: int, seed: int
) -> list[float]:
    if not grouped or any(not values for values in grouped.values()):
        raise ValueError("grouped bootstrap requires non-empty groups")
    song_sums = torch.tensor(
        [math.fsum(values) for values in grouped.values()], dtype=torch.float64
    )
    song_counts = torch.tensor(
        [len(values) for values in grouped.values()], dtype=torch.float64
    )
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        len(song_sums),
        (bootstrap_samples, len(song_sums)),
        generator=generator,
    )
    interval = torch.quantile(
        song_sums[draws].sum(1) / song_counts[draws].sum(1),
        torch.tensor((0.025, 0.975), dtype=torch.float64),
    )
    return [float(value) for value in interval]


@torch.inference_mode()
def _evaluate_harp(
    samples: list[dict],
    dataset: FeatureDataset,
    id_to_index: dict[str, int],
    noise: Tensor,
    model: HARPCore,
    system: HARPFlow,
    dct: Tensor,
    vocoder,
    config: HARPConfig,
    device: torch.device,
    condition: str,
    batch_size: int,
    steps: int,
    method: str,
) -> dict[str, dict[str, object]]:
    result = {}
    for length in LENGTHS:
        rows = []
        for offset in range(0, len(samples), batch_size):
            items = samples[offset : offset + batch_size]
            loaded = []
            for item in items:
                sample = dataset[
                    SampleRequest(
                        id_to_index[str(item["entry_id"])],
                        length,
                        int(item["ordinal"]),
                        int(item["start_frame"]),
                    )
                ]
                loaded.append(sample)
            cpu_batch = collate_features(loaded)
            batch = {name: value.to(device) for name, value in cpu_batch.items()}
            speaker = batch["speaker"]
            if condition == "null":
                speaker = torch.full_like(speaker, model.null_speaker_id)
            prediction = system.sample(
                batch["content"],
                batch["f0"],
                batch["rms"],
                speaker,
                batch["mask"],
                steps=steps,
                method=method,
                guidance_strength=1.0,
                initial_noise=noise[offset : offset + len(items), :length].to(device),
            ).cpu()
            for index, item in enumerate(items):
                row = {
                    "ordinal": int(item["ordinal"]),
                    "entry_id": str(item["entry_id"]),
                    **sample_metrics(
                        prediction[index],
                        cpu_batch["mel"][index],
                        cpu_batch["rms"][index],
                        cpu_batch["mask"][index],
                        dct,
                    ),
                }
                if vocoder is not None:
                    frames = int(cpu_batch["length"][index])
                    waveform = synthesize_pc_nsf(
                        vocoder,
                        prediction[index, :frames],
                        cpu_batch["f0"][index, :frames, 0],
                        device,
                    )
                    row["waveform"] = waveform_dbfs(waveform)
                rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "shadow_128_progress",
                        "condition": condition,
                        "frames": length,
                        "completed": min(offset + batch_size, len(samples)),
                        "total": len(samples),
                    }
                ),
                flush=True,
            )
        result[str(length)] = {"samples": rows, **aggregate_samples(rows)}
    return result


def aggregate_samples(rows: list[dict]) -> dict[str, object]:
    active = [
        float(row["active_raw_mse"])
        for row in rows
        if row["active_raw_mse"] is not None
    ]
    full = [float(row["full_raw_mse"]) for row in rows]
    silence = [
        float(row["silence_raw_mse"])
        for row in rows
        if row["silence_raw_mse"] is not None
    ]
    result: dict[str, object] = {
        "active_samples": len(active),
        "mean_active_raw_mse": statistics.fmean(active),
        "median_active_raw_mse": statistics.median(active),
        "mean_full_raw_mse": statistics.fmean(full),
        "mean_silence_raw_mse": statistics.fmean(silence) if silence else None,
        "catastrophe_count": sum(bool(row["catastrophe"]) for row in rows),
    }
    for key in (
        *[f"{name}_mse" for name in BANDS],
        "dct_32_127_corr",
        "dct_32_127_rms_ratio",
    ):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[f"mean_{key}"] = statistics.fmean(values) if values else None
    waveforms = [row["waveform"] for row in rows if "waveform" in row]
    if waveforms:
        finite_waveforms = [item for item in waveforms if item["finite"]]
        result["waveform"] = {
            "nonfinite_count": len(waveforms) - len(finite_waveforms),
            "mean_rms_dbfs": (
                statistics.fmean(item["rms_dbfs"] for item in finite_waveforms)
                if finite_waveforms
                else None
            ),
            "maximum_peak_dbfs": (
                max(item["peak_dbfs"] for item in finite_waveforms)
                if finite_waveforms
                else None
            ),
        }
    return result


def _load_v3_baseline(baseline: dict, samples: list[dict]) -> dict[str, dict]:
    expected = {(int(item["ordinal"]), str(item["entry_id"])) for item in samples}
    result = {}
    for length in LENGTHS:
        rows = baseline["models"]["v3_null"][str(length)]["samples"]
        if {_sample_key(item) for item in rows} != expected:
            raise ValueError(f"V3 baseline sample identity differs at {length}")
        normalized = []
        for row in rows:
            item = dict(row)
            full_mse = float(item["full_raw_mse"])
            item["catastrophe"] = full_mse > float(
                baseline["protocol"]["catastrophe_threshold_raw_mse"]
            )
            normalized.append(item)
        result[str(length)] = {"samples": normalized, **aggregate_samples(normalized)}
    return result


def _validate_protocol(
    panel: dict,
    baseline: dict,
    panel_path: Path,
    manifest_path: Path,
    speaker_to_id: dict[str, int],
) -> None:
    protocol = baseline.get("protocol", {})
    expected = {
        "frames": list(LENGTHS),
        "intervals": 32,
        "noise_seed": 20260904,
        "silence_threshold_rms": 0.001,
        "catastrophe_threshold_raw_mse": 5.0,
        "bootstrap": "dataset-song grouped",
    }
    mismatched = [key for key, value in expected.items() if protocol.get(key) != value]
    if mismatched:
        raise ValueError(f"historical V3 protocol differs in {mismatched}")
    if protocol.get("panel_lock_sha256") != _sha256(panel_path):
        raise ValueError("historical V3 baseline names a different panel lock")
    if panel.get("source", {}).get("manifest_sha256") != manifest_sha256(manifest_path):
        raise ValueError("Shadow-128 panel names a different manifest")
    if panel.get("source", {}).get("speaker_to_id_sha256") != _mapping_sha256(
        speaker_to_id
    ):
        raise ValueError("HARP speaker mapping differs from the Shadow-128 lock")
    if len(panel.get("samples", [])) != 128:
        raise ValueError("Shadow-128 panel must contain exactly 128 samples")


def _validate_samples(
    samples: list[dict], dataset: FeatureDataset, id_to_index: dict[str, int]
) -> None:
    if [int(item["ordinal"]) for item in samples] != list(range(128)):
        raise ValueError("Shadow-128 ordinals are not exactly 0..127")
    for item in samples:
        entry_id = str(item["entry_id"])
        if entry_id not in id_to_index:
            raise ValueError(f"Shadow-128 entry missing from manifest: {entry_id}")
        entry = dataset.entries[id_to_index[entry_id]]
        validate_panel_features(entry, item)
        if entry.dataset != item["dataset"] or entry.song != item["song"]:
            raise ValueError(f"Shadow-128 identity changed: {entry_id}")
        if int(item["maximum_frames"]) != max(LENGTHS):
            raise ValueError(f"Shadow-128 maximum length changed: {entry_id}")


def _selected_mse(error: Tensor, selection: Tensor) -> float | None:
    values = error[selection]
    return float(values.mean()) if values.numel() else None


def _correlation(left: Tensor, right: Tensor) -> float | None:
    if left.numel() < 2:
        return None
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator) if denominator > 0 else None


def _rms_ratio(prediction: Tensor, target: Tensor) -> float | None:
    if not prediction.numel():
        return None
    denominator = target.float().square().mean().sqrt()
    if denominator <= 0:
        return None
    return float(prediction.float().square().mean().sqrt() / denominator)


def _sample_key(item: dict) -> tuple[int, str]:
    return int(item["ordinal"]), str(item["entry_id"])


def _summary(payload: dict) -> dict:
    return {
        "checkpoint_progress": payload["checkpoint_progress"],
        "primary": {
            model: {
                length: {
                    "mean": values["mean_active_raw_mse"],
                    "median": values["median_active_raw_mse"],
                }
                for length, values in by_length.items()
            }
            for model, by_length in payload["models"].items()
        },
        "comparisons": payload["comparisons"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_sha256(mapping: dict[str, int]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
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


if __name__ == "__main__":
    main()
