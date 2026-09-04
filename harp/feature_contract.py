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

from .config import FeatureConfig, HarmonicConfig, ModelConfig


def slaney_mel_centers(channels: int, fmin: float, fmax: float) -> Tensor:
    """Return the internal filter peaks of the exact Slaney mel grid."""
    if channels <= 0 or not 0 <= fmin < fmax:
        raise ValueError("invalid Slaney mel frequency contract")
    boundaries = _mel_to_hz(
        torch.linspace(
            _hz_to_mel(fmin),
            _hz_to_mel(fmax),
            channels + 2,
            dtype=torch.float64,
        )
    )
    return boundaries[1:-1].float()


def tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy().astype(np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass
class FeatureContract:
    mel_center_hz: Tensor
    harmonic_mean: Tensor
    harmonic_std: Tensor
    rms_floor: float
    rms_log_mean: float
    rms_log_std: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        channels = self.mel_center_hz.numel()
        if self.mel_center_hz.shape != (channels,) or channels < 2:
            raise ValueError("mel centers must be a nontrivial vector")
        if self.harmonic_mean.shape != (4,) or self.harmonic_std.shape != (4,):
            raise ValueError("harmonic normalization must have four channels")
        tensors = (self.mel_center_hz, self.harmonic_mean, self.harmonic_std)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("feature contract contains non-finite tensors")
        if not bool((self.mel_center_hz[1:] > self.mel_center_hz[:-1]).all()):
            raise ValueError("mel centers must be strictly increasing")
        if bool((self.harmonic_std <= 0).any()):
            raise ValueError("harmonic standard deviations must be positive")
        scalars = (self.rms_floor, self.rms_log_mean, self.rms_log_std)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("feature contract contains non-finite RMS statistics")
        if self.rms_floor <= 0 or self.rms_log_std <= 0:
            raise ValueError("RMS floor and standard deviation must be positive")
        expected_hash = tensor_sha256(self.mel_center_hz)
        recorded_hash = self.metadata.get("mel_center_sha256")
        if recorded_hash is not None and recorded_hash != expected_hash:
            raise ValueError("mel center vector hash does not match metadata")

    @property
    def channels(self) -> int:
        return self.mel_center_hz.numel()

    def to_payload(self) -> dict[str, Any]:
        return {
            "mel_center_hz": self.mel_center_hz.detach().cpu(),
            "harmonic_mean": self.harmonic_mean.detach().cpu(),
            "harmonic_std": self.harmonic_std.detach().cpu(),
            "rms_floor": self.rms_floor,
            "rms_log_mean": self.rms_log_mean,
            "rms_log_std": self.rms_log_std,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata["mel_center_sha256"] = tensor_sha256(self.mel_center_hz)
        np.savez(
            target,
            mel_center_hz=self.mel_center_hz.cpu().numpy().astype(np.float32),
            harmonic_mean=self.harmonic_mean.cpu().numpy().astype(np.float32),
            harmonic_std=self.harmonic_std.cpu().numpy().astype(np.float32),
            rms_floor=np.array(self.rms_floor, dtype=np.float64),
            rms_log_mean=np.array(self.rms_log_mean, dtype=np.float64),
            rms_log_std=np.array(self.rms_log_std, dtype=np.float64),
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )
        return hashlib.sha256(target.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> FeatureContract:
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                mel_center_hz=torch.from_numpy(payload["mel_center_hz"].copy()),
                harmonic_mean=torch.from_numpy(payload["harmonic_mean"].copy()),
                harmonic_std=torch.from_numpy(payload["harmonic_std"].copy()),
                rms_floor=float(payload["rms_floor"]),
                rms_log_mean=float(payload["rms_log_mean"]),
                rms_log_std=float(payload["rms_log_std"]),
                metadata=json.loads(str(payload["metadata"].item())),
            )


def neutral_feature_contract(
    channels: int, fmin: float, fmax: float, *, rms_floor: float = 1e-5
) -> FeatureContract:
    """Construct a neutral contract for unit tests and synthetic smoke only."""
    centers = slaney_mel_centers(channels, fmin, fmax)
    return FeatureContract(
        mel_center_hz=centers,
        harmonic_mean=torch.zeros(4),
        harmonic_std=torch.ones(4),
        rms_floor=rms_floor,
        rms_log_mean=0.0,
        rms_log_std=1.0,
        metadata={
            "artifact_type": "synthetic_feature_contract_only",
            "mel_center_sha256": tensor_sha256(centers),
        },
    )


def validate_feature_contract(
    contract: FeatureContract,
    model: ModelConfig,
    harmonic: HarmonicConfig,
    feature: FeatureConfig,
) -> None:
    expected_centers = slaney_mel_centers(
        model.mel_channels, harmonic.fmin, harmonic.fmax
    )
    if not torch.equal(contract.mel_center_hz.cpu(), expected_centers):
        raise ValueError("feature contract mel centers differ from the Slaney frontend")
    expected_metadata = {
        "mel_scale": feature.mel_scale,
        "mel_norm": feature.mel_norm,
        "hop_length": feature.hop_length,
        "n_fft": feature.n_fft,
        "win_length": feature.win_length,
        "power": feature.power,
        "center": feature.center,
        "pad_mode": feature.pad_mode,
        "log_base": feature.log_base,
        "log_clamp": feature.log_clamp,
        "mel_fmin": harmonic.fmin,
        "mel_fmax": harmonic.fmax,
        "sample_rate": harmonic.sample_rate,
        "f0_min": harmonic.f0_min,
        "f0_max": harmonic.f0_max,
        "nyquist_ratio": harmonic.nyquist_ratio,
        "narrow_bandwidth_semitones": harmonic.narrow_bandwidth_semitones,
        "wide_bandwidth_semitones": harmonic.wide_bandwidth_semitones,
        "harmonic_weighting": "n^-0.5",
        "occupancy_normalization": "per_frame_max",
        "distance_definition": "signed_semitones_clipped_6_divided_6",
        "index_definition": "log1p_n_divided_log1p_global_nmax",
        "waveform_amplitude_convention": feature.waveform_amplitude_convention,
        "rms_definition": feature.rms_definition,
    }
    mismatched = [
        name
        for name, expected in expected_metadata.items()
        if contract.metadata.get(name) != expected
    ]
    if contract.rms_floor != feature.rms_floor:
        mismatched.append("rms_floor")
    if mismatched:
        raise ValueError(f"feature contract differs from config in {mismatched}")


class FeatureStatisticsAccumulator:
    def __init__(self, harmonic_features: torch.nn.Module, rms_floor: float) -> None:
        if rms_floor <= 0:
            raise ValueError("RMS floor must be positive")
        self.harmonic_features = harmonic_features
        self.rms_floor = rms_floor
        self.harmonic_sum = torch.zeros(4, dtype=torch.float64)
        self.harmonic_square_sum = torch.zeros(4, dtype=torch.float64)
        self.harmonic_count = 0
        self.rms_sum = 0.0
        self.rms_square_sum = 0.0
        self.rms_count = 0

    @torch.inference_mode()
    def update(self, f0: Tensor, rms: Tensor, mask: Tensor) -> None:
        valid = mask.bool()
        voiced = valid & torch.isfinite(f0[..., 0]) & (f0[..., 0] > 0)
        harmonic = self.harmonic_features(f0, voiced[..., None])
        selected = harmonic.movedim(-2, -1)[voiced].reshape(-1, 4).double().cpu()
        if selected.numel():
            self.harmonic_sum += selected.sum(dim=0)
            self.harmonic_square_sum += selected.square().sum(dim=0)
            self.harmonic_count += selected.shape[0]
        log_rms = torch.log(rms[..., 0].float().clamp_min(0) + self.rms_floor)
        selected_rms = log_rms[valid].double().cpu()
        self.rms_sum += float(selected_rms.sum())
        self.rms_square_sum += float(selected_rms.square().sum())
        self.rms_count += selected_rms.numel()

    def finalize(
        self, mel_center_hz: Tensor, metadata: dict[str, Any]
    ) -> FeatureContract:
        if self.harmonic_count < 2 or self.rms_count < 2:
            raise ValueError("insufficient samples for feature statistics")
        harmonic_mean = self.harmonic_sum / self.harmonic_count
        harmonic_variance = (
            self.harmonic_square_sum - self.harmonic_count * harmonic_mean.square()
        ) / (self.harmonic_count - 1)
        rms_mean = self.rms_sum / self.rms_count
        rms_variance = (self.rms_square_sum - self.rms_count * rms_mean * rms_mean) / (
            self.rms_count - 1
        )
        resolved_metadata = {
            **metadata,
            "artifact_type": "feature_contract_v1",
            "harmonic_statistic_values": self.harmonic_count,
            "rms_statistic_frames": self.rms_count,
            "mel_center_sha256": tensor_sha256(mel_center_hz),
        }
        return FeatureContract(
            mel_center_hz=mel_center_hz.float(),
            harmonic_mean=harmonic_mean.float(),
            harmonic_std=harmonic_variance.clamp_min(1e-12).sqrt().float(),
            rms_floor=self.rms_floor,
            rms_log_mean=rms_mean,
            rms_log_std=math.sqrt(max(rms_variance, 1e-12)),
            metadata=resolved_metadata,
        )


def _hz_to_mel(frequency: float) -> float:
    linear_spacing = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / linear_spacing
    log_step = math.log(6.4) / 27.0
    if frequency >= min_log_hz:
        return min_log_mel + math.log(frequency / min_log_hz) / log_step
    return frequency / linear_spacing


def _mel_to_hz(mel: Tensor) -> Tensor:
    linear_spacing = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / linear_spacing
    log_step = math.log(6.4) / 27.0
    frequency = mel * linear_spacing
    logarithmic = mel >= min_log_mel
    frequency[logarithmic] = min_log_hz * torch.exp(
        log_step * (mel[logarithmic] - min_log_mel)
    )
    return frequency
