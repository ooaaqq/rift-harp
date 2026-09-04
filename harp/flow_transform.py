from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

ALPHA_CANDIDATES = (0.5, 0.625, 0.75, 0.875, 1.0)


@dataclass(frozen=True)
class CovarianceDiagnostics:
    offdiag_ratio: float
    max_abs_corr: float
    variance_p95_p05: float
    variance_max_min: float


@dataclass
class FlowTransform:
    mean: Tensor
    basis: Tensor
    gain: Tensor
    lambda_raw: Tensor
    lambda_effective: Tensor
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        channels = self.mean.numel()
        expected = (channels, channels)
        if self.basis.shape != expected:
            raise ValueError(f"basis must have shape {expected}")
        for name in ("gain", "lambda_raw", "lambda_effective"):
            if getattr(self, name).shape != (channels,):
                raise ValueError(f"{name} must have shape ({channels},)")
        if not all(
            torch.isfinite(value).all()
            for value in (
                self.mean,
                self.basis,
                self.gain,
                self.lambda_raw,
                self.lambda_effective,
            )
        ):
            raise ValueError("flow transform contains non-finite values")
        if (self.gain <= 0).any() or (self.lambda_effective <= 0).any():
            raise ValueError("gain and lambda must be positive")

    @property
    def channels(self) -> int:
        return self.mean.numel()

    def transform(self, mel: Tensor) -> Tensor:
        mean = self.mean.to(device=mel.device, dtype=torch.float32)
        basis = self.basis.to(device=mel.device, dtype=torch.float32)
        gain = self.gain.to(device=mel.device, dtype=torch.float32)
        return ((mel.float() - mean) @ basis.T) * gain

    def inverse(self, transformed: Tensor) -> Tensor:
        mean = self.mean.to(device=transformed.device, dtype=torch.float32)
        basis = self.basis.to(device=transformed.device, dtype=torch.float32)
        gain = self.gain.to(device=transformed.device, dtype=torch.float32)
        return (transformed.float() / gain) @ basis + mean

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "mean": self.mean.cpu().numpy().astype(np.float32),
            "basis": self.basis.cpu().numpy().astype(np.float32),
            "gain": self.gain.cpu().numpy().astype(np.float32),
            "lambda_raw": self.lambda_raw.cpu().numpy().astype(np.float32),
            "lambda_effective": self.lambda_effective.cpu().numpy().astype(np.float32),
            "metadata": np.array(json.dumps(self.metadata, sort_keys=True)),
        }
        np.savez(target, **arrays)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return digest

    @classmethod
    def load(cls, path: str | Path) -> FlowTransform:
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                mean=torch.from_numpy(payload["mean"].copy()),
                basis=torch.from_numpy(payload["basis"].copy()),
                gain=torch.from_numpy(payload["gain"].copy()),
                lambda_raw=torch.from_numpy(payload["lambda_raw"].copy()),
                lambda_effective=torch.from_numpy(payload["lambda_effective"].copy()),
                metadata=json.loads(str(payload["metadata"].item())),
            )


def orthonormal_dct(channels: int) -> Tensor:
    indices = torch.arange(channels, dtype=torch.float64)
    modes = torch.arange(channels, dtype=torch.float64)[:, None]
    basis = torch.cos(math.pi / channels * (indices + 0.5) * modes)
    basis[0] *= math.sqrt(1.0 / channels)
    basis[1:] *= math.sqrt(2.0 / channels)
    return basis.float()


def fit_flow_transform(
    frames: Tensor,
    *,
    seed: int,
    validation_fraction: float = 0.2,
    lambda_floor: float = 1e-4,
    offdiag_limit: float = 0.10,
    max_corr_limit: float = 0.30,
    sampler_hash: str,
    dataset_manifest_hash: str,
    crop_policy_hash: str,
) -> FlowTransform:
    if frames.ndim != 2 or frames.shape[0] < 10 or frames.shape[1] < 2:
        raise ValueError("frames must be [N,C] with at least 10 rows and 2 channels")
    if not torch.isfinite(frames).all():
        raise ValueError("statistics frames contain non-finite values")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(frames.shape[0], generator=generator)
    validation_count = max(1, round(frames.shape[0] * validation_fraction))
    validation = frames[order[:validation_count]].double()
    fitting = frames[order[validation_count:]].double()
    mean = fitting.mean(dim=0)
    centered_fit = fitting - mean
    centered_validation = validation - mean

    dct = orthonormal_dct(frames.shape[1]).double()
    dct_fit = centered_fit @ dct.T
    dct_validation = centered_validation @ dct.T
    dct_val_diagnostics = covariance_diagnostics(dct_validation)
    if (
        dct_val_diagnostics.offdiag_ratio <= offdiag_limit
        and dct_val_diagnostics.max_abs_corr <= max_corr_limit
    ):
        basis = dct
        basis_kind = "dct"
        rotated_fit = dct_fit
        rotated_validation = dct_validation
    else:
        covariance = _covariance(centered_fit)
        _, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors.flip(1).T
        basis_kind = "pca"
        rotated_fit = centered_fit @ basis.T
        rotated_validation = centered_validation @ basis.T

    variance = rotated_fit.var(dim=0, correction=1).clamp_min(
        torch.finfo(torch.float64).eps
    )
    try:
        alpha, gain, lambda_raw = _select_gain(variance)
    except ValueError as error:
        raise ValueError(
            f"basis={basis_kind}; dct_validation="
            f"{_diagnostics_dict(dct_val_diagnostics)}; {error}"
        ) from error
    transformed_fit = rotated_fit * gain
    transformed_validation = rotated_validation * gain
    fit_diagnostics = covariance_diagnostics(transformed_fit)
    validation_diagnostics = covariance_diagnostics(transformed_validation)
    if (
        validation_diagnostics.offdiag_ratio > offdiag_limit
        or validation_diagnostics.max_abs_corr > max_corr_limit
    ):
        raise ValueError(
            "flow transform contract failed validation correlation criteria: "
            f"rho={validation_diagnostics.offdiag_ratio:.4f}, "
            f"max_corr={validation_diagnostics.max_abs_corr:.4f}"
        )
    if (
        validation_diagnostics.variance_p95_p05 > 16
        or validation_diagnostics.variance_max_min > 64
    ):
        raise ValueError("flow transform contract failed validation variance spread")
    lambda_effective = lambda_raw.clamp_min(lambda_floor)
    floor_fraction = float((lambda_raw < lambda_floor).double().mean())
    metadata = {
        "artifact_type": "flow_transform_v1",
        "contract_accepted": True,
        "basis_kind": basis_kind,
        "alpha": alpha,
        "gain_clip_relative_median": [1 / 3, 3],
        "lambda_floor": lambda_floor,
        "lambda_floor_fraction": floor_fraction,
        "seed": seed,
        "frame_count": int(frames.shape[0]),
        "fit_frame_count": int(fitting.shape[0]),
        "validation_frame_count": int(validation.shape[0]),
        "sampler_hash": sampler_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "crop_policy_hash": crop_policy_hash,
        "dct_validation": _diagnostics_dict(dct_val_diagnostics),
        "fit": _diagnostics_dict(fit_diagnostics),
        "validation": _diagnostics_dict(validation_diagnostics),
    }
    return FlowTransform(
        mean=mean.float(),
        basis=basis.float(),
        gain=gain.float(),
        lambda_raw=lambda_raw.float(),
        lambda_effective=lambda_effective.float(),
        metadata=metadata,
    )


def covariance_diagnostics(samples: Tensor) -> CovarianceDiagnostics:
    covariance = _covariance(samples.double())
    diagonal = covariance.diag().clamp_min(torch.finfo(torch.float64).eps)
    offdiag = covariance - torch.diag_embed(diagonal)
    ratio = offdiag.norm() / covariance.norm().clamp_min(torch.finfo(torch.float64).eps)
    scale = diagonal.sqrt()
    correlation = covariance / (scale[:, None] * scale[None, :])
    correlation.fill_diagonal_(0)
    sorted_variance = diagonal.sort().values
    p05_index = max(0, math.floor(0.05 * (len(sorted_variance) - 1)))
    p95_index = min(
        len(sorted_variance) - 1, math.ceil(0.95 * (len(sorted_variance) - 1))
    )
    return CovarianceDiagnostics(
        offdiag_ratio=float(ratio),
        max_abs_corr=float(correlation.abs().max()),
        variance_p95_p05=float(sorted_variance[p95_index] / sorted_variance[p05_index]),
        variance_max_min=float(sorted_variance[-1] / sorted_variance[0]),
    )


def _select_gain(variance: Tensor) -> tuple[float, Tensor, Tensor]:
    epsilon = torch.finfo(variance.dtype).eps
    failures = []
    for alpha in ALPHA_CANDIDATES:
        raw = (variance + epsilon).pow(-alpha / 2)
        relative = (raw / raw.median()).clamp(1 / 3, 3)
        global_scale = (relative.square() * variance).median().rsqrt()
        gain = global_scale * relative
        transformed_variance = gain.square() * variance
        ordered = transformed_variance.sort().values
        p05 = ordered[max(0, math.floor(0.05 * (len(ordered) - 1)))]
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * (len(ordered) - 1)))]
        p95_p05 = float(p95 / p05)
        max_min = float(ordered[-1] / ordered[0])
        failures.append(
            {
                "alpha": alpha,
                "p95_p05": p95_p05,
                "max_min": max_min,
                "min_mode": int(transformed_variance.argmin()),
                "max_mode": int(transformed_variance.argmax()),
                "gain_clip_low_fraction": float((relative == 1 / 3).double().mean()),
                "gain_clip_high_fraction": float((relative == 3).double().mean()),
            }
        )
        if p95_p05 <= 16 and max_min <= 64:
            return alpha, gain, transformed_variance
    raise ValueError(
        f"no alpha satisfies variance spread under the 3x gain cap: {failures}"
    )


def _covariance(samples: Tensor) -> Tensor:
    centered = samples - samples.mean(dim=0)
    return centered.T @ centered / max(1, samples.shape[0] - 1)


def _diagnostics_dict(value: CovarianceDiagnostics) -> dict[str, float]:
    return {
        "offdiag_ratio": value.offdiag_ratio,
        "max_abs_corr": value.max_abs_corr,
        "variance_p95_p05": value.variance_p95_p05,
        "variance_max_min": value.variance_max_min,
    }
