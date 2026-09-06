from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from .checkpoint import validate_checkpoint_contract, validate_external_artifacts
from .config import HARPConfig
from .data import FeatureDataset, SampleRequest, collate_features
from .feature_contract import FeatureContract
from .flow import HARPFlow, flow_coefficients
from .flow_transform import FlowTransform, full_precision_matmul, orthonormal_dct
from .manifest import load_manifest, manifest_sha256
from .model import HARPCore
from .panel_cli import validate_panel_features
from .performance import configure_cuda

TIMESTEPS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
BANDS = {"dct_0_15": (0, 16), "dct_16_31": (16, 32), "dct_32_127": (32, 128)}


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.pred_sum = 0.0
        self.target_sum = 0.0
        self.pred_square_sum = 0.0
        self.target_square_sum = 0.0
        self.cross_sum = 0.0
        self.error_square_sum = 0.0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        prediction = prediction.double()
        target = target.double()
        self.count += prediction.numel()
        self.pred_sum += float(prediction.sum())
        self.target_sum += float(target.sum())
        self.pred_square_sum += float(prediction.square().sum())
        self.target_square_sum += float(target.square().sum())
        self.cross_sum += float((prediction * target).sum())
        self.error_square_sum += float((prediction - target).square().sum())

    def report(self) -> dict[str, float | int]:
        if self.count < 2:
            return {"count": self.count}
        pred_mean = self.pred_sum / self.count
        target_mean = self.target_sum / self.count
        pred_var = self.pred_square_sum / self.count - pred_mean**2
        target_var = self.target_square_sum / self.count - target_mean**2
        covariance = self.cross_sum / self.count - pred_mean * target_mean
        mse = self.error_square_sum / self.count
        return {
            "count": self.count,
            "mse": mse,
            "nmse": mse / max(self.target_square_sum / self.count, 1e-12),
            "correlation": covariance / math.sqrt(max(pred_var * target_var, 1e-24)),
            "prediction_to_target_rms": math.sqrt(
                self.pred_square_sum / max(self.target_square_sum, 1e-24)
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="HARP raw-mel local field audit")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
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
    panel_artifact = json.loads(args.panels.read_text())
    if panel_artifact.get("manifest_sha256") != manifest_sha256(args.manifest):
        raise ValueError("fixed panel manifest hash differs")
    entries = load_manifest(args.manifest)
    id_to_index = {entry.id: index for index, entry in enumerate(entries)}
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        speaker_to_id=checkpoint["speaker_to_id"],
        voiced_crop_probability=config.training.voiced_crop_probability,
    )
    loaded_panels = {}
    for name, items in panel_artifact["panels"].items():
        loaded_panels[name] = []
        for item in items:
            entry = entries[id_to_index[item["entry_id"]]]
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
            loaded_panels[name].append(sample)
    thresholds = _stratum_thresholds(loaded_panels)
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
    system = HARPFlow(
        model,
        transform,
        speaker_drop_probability=0,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    ).eval()
    dct = orthonormal_dct(config.model.mel_channels).to(device)
    accumulators: dict[tuple[str, str, float, str, str], MetricAccumulator] = (
        defaultdict(MetricAccumulator)
    )
    with torch.inference_mode():
        for state_name in ("raw", "ema"):
            model.load_state_dict(
                checkpoint["model" if state_name == "raw" else "ema"], strict=True
            )
            model.eval()
            for panel_name, samples in loaded_panels.items():
                for sample in samples:
                    batch = {
                        name: value.to(device)
                        for name, value in collate_features([sample]).items()
                    }
                    target = transform.transform(batch["mel"])
                    generator = torch.Generator(device=device).manual_seed(
                        int(sample["noise_seed"])
                    )
                    noise = torch.randn(
                        target.shape, device=device, generator=generator
                    )
                    harmonic = model.prepare_harmonic(batch["f0"])
                    rms = model.prepare_rms(batch["rms"])
                    selections = _strata(batch, thresholds)
                    for timestep_value in TIMESTEPS:
                        timestep = torch.full(
                            (target.shape[0],), timestep_value, device=device
                        )
                        state = (1 - timestep_value) * noise + timestep_value * target
                        coefficients = flow_coefficients(
                            timestep,
                            transform.lambda_raw,
                            lambda_floor=config.flow.lambda_floor,
                            q_floor=config.flow.q_floor,
                        )
                        residual = system._model_residual(
                            coefficients.c_in * state,
                            batch["content"],
                            batch["f0"],
                            rms,
                            harmonic,
                            batch["speaker"],
                            timestep,
                            batch["mask"],
                        )
                        prediction_y = (
                            coefficients.c_skip * state + coefficients.c_out * residual
                        )
                        target_y = target - noise
                        prediction_raw = full_precision_matmul(
                            prediction_y / transform.gain, transform.basis
                        )
                        target_raw = full_precision_matmul(
                            target_y / transform.gain, transform.basis
                        )
                        prediction_dct = full_precision_matmul(prediction_raw, dct.T)
                        target_dct = full_precision_matmul(target_raw, dct.T)
                        for stratum, selection in selections.items():
                            for band, (start, end) in BANDS.items():
                                accumulators[
                                    (
                                        state_name,
                                        panel_name,
                                        timestep_value,
                                        stratum,
                                        band,
                                    )
                                ].update(
                                    prediction_dct[..., start:end][selection],
                                    target_dct[..., start:end][selection],
                                )
    results = [
        {
            "state": key[0],
            "panel": key[1],
            "timestep": key[2],
            "stratum": key[3],
            "band": key[4],
            **value.report(),
        }
        for key, value in sorted(accumulators.items())
    ]
    payload = {
        "artifact_type": "harp_local_field_audit_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_progress": checkpoint["progress"],
        "panel_artifact": str(args.panels),
        "raw_velocity_inverse": "v_x=(v_y/gain)@basis; no mean",
        "stratum_thresholds": thresholds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _stratum_thresholds(
    panels: dict[str, list[dict[str, Tensor]]],
) -> dict[str, list[float] | float]:
    f0_values = []
    deltas = []
    for samples in panels.values():
        for sample in samples:
            f0 = sample["f0"][..., 0]
            voiced = torch.isfinite(f0) & (f0 > 0)
            f0_values.append(f0[voiced])
            log_f0 = torch.where(voiced, torch.log2(f0.clamp_min(1)), torch.nan)
            delta = (log_f0[1:] - log_f0[:-1]).abs()
            valid_delta = voiced[1:] & voiced[:-1]
            deltas.append(delta[valid_delta])
    f0 = torch.cat(f0_values)
    delta = torch.cat(deltas)
    return {
        "f0_quartile_hz": torch.quantile(f0, torch.tensor([0.25, 0.5, 0.75])).tolist(),
        "stable_delta_log2_max": float(torch.quantile(delta, 0.25)),
        "rapid_delta_log2_min": float(torch.quantile(delta, 0.75)),
    }


def _strata(
    batch: dict[str, Tensor], thresholds: dict[str, list[float] | float]
) -> dict[str, Tensor]:
    mask = batch["mask"]
    f0 = batch["f0"][..., 0]
    voiced = mask & torch.isfinite(f0) & (f0 > 0)
    boundaries = torch.tensor(
        thresholds["f0_quartile_hz"], device=f0.device, dtype=f0.dtype
    )
    log_f0 = torch.where(voiced, torch.log2(f0.clamp_min(1)), torch.zeros_like(f0))
    delta = torch.zeros_like(log_f0)
    delta[:, 1:] = (log_f0[:, 1:] - log_f0[:, :-1]).abs()
    adjacent_voiced = voiced.clone()
    adjacent_voiced[:, 1:] &= voiced[:, :-1]
    return {
        "all": mask,
        "voiced": voiced,
        "unvoiced": mask & ~voiced,
        "f0_q1": voiced & (f0 <= boundaries[0]),
        "f0_q2": voiced & (f0 > boundaries[0]) & (f0 <= boundaries[1]),
        "f0_q3": voiced & (f0 > boundaries[1]) & (f0 <= boundaries[2]),
        "f0_q4": voiced & (f0 > boundaries[2]),
        "stable_f0": adjacent_voiced
        & (delta <= float(thresholds["stable_delta_log2_max"])),
        "rapid_f0": adjacent_voiced
        & (delta >= float(thresholds["rapid_delta_log2_min"])),
    }


if __name__ == "__main__":
    main()
