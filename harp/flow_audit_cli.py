from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .config import HARPConfig
from .contracts import exposure_semantics_hash, stats_run_hash
from .data import FeatureDataset, HierarchicalBatchSampler, collate_features
from .flow import flow_coefficients, sample_timestep
from .flow_transform import FlowTransform
from .manifest import load_manifest, manifest_sha256


@dataclass
class ModeMoments:
    count: int
    total: Tensor
    square_total: Tensor

    @classmethod
    def empty(cls, channels: int) -> ModeMoments:
        return cls(
            0,
            torch.zeros(channels, dtype=torch.float64),
            torch.zeros(channels, dtype=torch.float64),
        )

    def update(self, values: Tensor) -> None:
        if not values.numel():
            return
        values = values.float()
        self.count += values.shape[0]
        self.total += values.sum(dim=0).double().cpu()
        self.square_total += values.square().sum(dim=0).double().cpu()

    def report(self) -> dict[str, object]:
        if self.count < 2:
            return {"frames": self.count}
        mean = self.total / self.count
        second = self.square_total / self.count
        variance = (self.square_total - self.count * mean.square()) / (self.count - 1)
        return {
            "frames": self.count,
            "conditional_mean": mean.tolist(),
            "conditional_variance": variance.tolist(),
            "conditional_second_moment": second.tolist(),
            "variance_summary": _summary(variance),
            "second_moment_mode_mean": float(second.mean()),
        }


class CovarianceAccumulator:
    def __init__(self, channels: int) -> None:
        self.count = 0
        self.total = torch.zeros(channels, dtype=torch.float64)
        self.outer = torch.zeros(channels, channels, dtype=torch.float64)

    def update(self, values: Tensor) -> None:
        values = values.float()
        self.count += values.shape[0]
        self.total += values.sum(dim=0).double().cpu()
        # FP32 GEMM is accurate enough for the independent contract thresholds;
        # accumulation and the final covariance calculation remain FP64.
        self.outer += (values.T @ values).double().cpu()

    def covariance(self) -> Tensor:
        mean = self.total / self.count
        return (self.outer - self.count * mean[:, None] * mean[None, :]) / (
            self.count - 1
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent HARP flow audit")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transform", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=750_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 500_000 <= args.frames <= 1_000_000:
        raise ValueError("independent audit requires 0.5M to 1M valid frames")

    config = HARPConfig.load(args.config)
    manifest_hash = manifest_sha256(args.manifest)
    entries = [
        entry
        for entry in load_manifest(args.manifest)
        if entry.split == "train" and entry.quality_status == "accepted"
    ]
    audit_config = _with_seed(config, args.seed)
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        voiced_crop_probability=config.training.voiced_crop_probability,
        mel_only=True,
    )
    sampler = HierarchicalBatchSampler(entries, audit_config)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_features,
        num_workers=config.sampling.num_workers,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=config.sampling.prefetch_factor,
        persistent_workers=config.sampling.persistent_workers,
    )
    transform_path = args.transform or Path(config.flow.transform_path)
    transform_sha256 = _sha256(transform_path)
    device = torch.device(args.device)
    transform = FlowTransform.load(transform_path).to(device)
    gain_clip = transform.metadata.get("gain_clip_relative_median")
    if gain_clip not in (4.0, [0.25, 4.0]):
        raise ValueError("flow transform does not use the frozen 4x relative gain cap")

    iterator = iter(loader)
    calibration_f0 = []
    calibration_frames = 0
    while calibration_frames < 100_000:
        batch = next(iterator)
        selected = batch["f0"][..., 0][batch["mask"] & (batch["f0"][..., 0] > 0)]
        calibration_f0.append(selected)
        calibration_frames += int(batch["mask"].sum())
    quartiles = torch.quantile(
        torch.cat(calibration_f0), torch.tensor([0.25, 0.5, 0.75])
    )

    channels = transform.channels
    moments = {
        name: ModeMoments.empty(channels)
        for name in ("all", "voiced", "unvoiced", "f0_q1", "f0_q2", "f0_q3", "f0_q4")
    }
    covariance = CovarianceAccumulator(channels)
    seen_keys: set[tuple[int, int, int]] = set()
    accepted_frames = 0
    duplicate_crops = 0
    q_floor_hits = 0
    q_values = 0
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    while accepted_frames < args.frames:
        batch = next(iterator)
        keep_samples = torch.ones(batch["length"].shape[0], dtype=torch.bool)
        for index in range(keep_samples.numel()):
            key = (
                int(batch["entry_index"][index]),
                int(batch["crop_start"][index]),
                int(batch["length"][index]),
            )
            if key in seen_keys:
                keep_samples[index] = False
                duplicate_crops += 1
            else:
                seen_keys.add(key)
        mask = batch["mask"] & keep_samples[:, None]
        remaining = args.frames - accepted_frames
        if int(mask.sum()) > remaining:
            selected = mask.flatten().nonzero().flatten()[:remaining]
            limited = torch.zeros_like(mask.flatten())
            limited[selected] = True
            mask = limited.view_as(mask)

        mel = batch["mel"].to(device, non_blocking=True)
        device_mask = mask.to(device, non_blocking=True)
        target = transform.transform(mel)
        valid_target = target[device_mask]
        covariance.update(valid_target)
        timestep = sample_timestep(target.shape[0], device)
        noise = torch.randn(target.shape, device=device, dtype=torch.float32)
        expanded_t = timestep[:, None, None]
        state = (1 - expanded_t) * noise + expanded_t * target
        coefficients = flow_coefficients(
            timestep,
            transform.lambda_raw,
            lambda_floor=config.flow.lambda_floor,
            q_floor=config.flow.q_floor,
        )
        residual_target = (
            target - noise - coefficients.c_skip * state
        ) / coefficients.c_out
        valid_residual = residual_target[device_mask]
        valid_f0 = batch["f0"][..., 0][mask].to(device)
        voiced = torch.isfinite(valid_f0) & (valid_f0 > 0)
        moments["all"].update(valid_residual)
        moments["voiced"].update(valid_residual[voiced])
        moments["unvoiced"].update(valid_residual[~voiced])
        boundaries = quartiles.to(device)
        quartile_masks = (
            voiced & (valid_f0 <= boundaries[0]),
            voiced & (valid_f0 > boundaries[0]) & (valid_f0 <= boundaries[1]),
            voiced & (valid_f0 > boundaries[1]) & (valid_f0 <= boundaries[2]),
            voiced & (valid_f0 > boundaries[2]),
        )
        for name, selection in zip(
            ("f0_q1", "f0_q2", "f0_q3", "f0_q4"), quartile_masks, strict=True
        ):
            moments[name].update(valid_residual[selection])
        q_floor_hits += round(coefficients.q_floor_fraction * coefficients.c_in.numel())
        q_values += coefficients.c_in.numel()
        accepted_frames += int(mask.sum())

    covariance_report = _covariance_report(covariance.covariance())
    reports = {name: value.report() for name, value in moments.items()}
    all_energy = float(reports["all"]["second_moment_mode_mean"])
    for report in reports.values():
        if "second_moment_mode_mean" not in report:
            continue
        exposure_share = int(report["frames"]) / accepted_frames
        energy_share = (
            exposure_share * float(report["second_moment_mode_mean"]) / all_energy
        )
        report["exposure_share"] = exposure_share
        report["target_energy_share"] = energy_share
        report["energy_to_exposure_ratio"] = energy_share / max(exposure_share, 1e-12)

    roundtrip_source = torch.randn(32, 17, channels, device=device)
    roundtrip_error = (
        (transform.inverse(transform.transform(roundtrip_source)) - roundtrip_source)
        .abs()
        .max()
    )
    accepted = (
        covariance_report["offdiag_ratio"] <= 0.10
        and covariance_report["max_abs_corr"] <= 0.30
        and covariance_report["variance_p95_p05"] <= 16
        and covariance_report["variance_max_min"] <= 64
        and 0.8 <= covariance_report["variance_median"] <= 1.25
        and not bool((transform.lambda_raw < config.flow.lambda_floor).any())
        and q_floor_hits == 0
        and float(roundtrip_error) <= 1e-4
    )
    payload = {
        "artifact_type": "flow_transform_v1_independent_audit",
        "contract_accepted": accepted,
        "flow_transform_sha256": transform_sha256,
        "exposure_semantics_hash": exposure_semantics_hash(config, manifest_hash),
        "stats_run_hash": stats_run_hash(
            seed=args.seed, frame_count=args.frames, stream="independent_audit"
        ),
        "seed": args.seed,
        "frames": accepted_frames,
        "fit_stream_overlap_check": (
            "new seed and no duplicates within audit; legacy fit crop keys unavailable"
        ),
        "duplicate_audit_crops_skipped": duplicate_crops,
        "f0_quartile_hz": quartiles.tolist(),
        "covariance": covariance_report,
        "gain_clip_relative_median": transform.metadata.get(
            "gain_clip_relative_median"
        ),
        "gain_audit": transform.metadata.get("gain_audit"),
        "conditional_residual": reports,
        "lambda_floor_hit_fraction": float(
            (transform.lambda_raw < config.flow.lambda_floor).float().mean()
        ),
        "q_floor_hit_fraction": q_floor_hits / max(1, q_values),
        "roundtrip_max_abs_error": float(roundtrip_error),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not accepted:
        raise ValueError("independent flow transform audit rejected the contract")


def _with_seed(config: HARPConfig, seed: int) -> HARPConfig:
    import dataclasses

    return dataclasses.replace(
        config, sampling=dataclasses.replace(config.sampling, seed=seed)
    )


def _covariance_report(covariance: Tensor) -> dict[str, float]:
    diagonal = covariance.diag().clamp_min(torch.finfo(torch.float64).eps)
    offdiag = covariance - torch.diag_embed(diagonal)
    scale = diagonal.sqrt()
    correlation = covariance / (scale[:, None] * scale[None, :])
    correlation.fill_diagonal_(0)
    ordered = diagonal.sort().values
    p05 = ordered[math.floor(0.05 * (len(ordered) - 1))]
    p95 = ordered[math.ceil(0.95 * (len(ordered) - 1))]
    return {
        "offdiag_ratio": float(offdiag.norm() / covariance.norm()),
        "max_abs_corr": float(correlation.abs().max()),
        "variance_p95_p05": float(p95 / p05),
        "variance_max_min": float(ordered[-1] / ordered[0]),
        "variance_median": float(diagonal.median()),
    }


def _summary(values: Tensor) -> dict[str, float]:
    ordered = values.sort().values
    return {
        "median": float(ordered.median()),
        "p95": float(ordered[math.ceil(0.95 * (len(ordered) - 1))]),
        "max": float(ordered[-1]),
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
