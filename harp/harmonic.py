from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def mel_center_frequencies(
    channels: int, fmin: float, fmax: float, *, device: torch.device | None = None
) -> Tensor:
    if channels <= 0 or not 0 < fmin < fmax:
        raise ValueError("invalid mel frequency contract")
    mel_min = 2595.0 * math.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + fmax / 700.0)
    mel = torch.linspace(mel_min, mel_max, channels, device=device)
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


class HarmonicFeatures(nn.Module):
    """Fixed harmonic geometry; it carries positions, never target amplitudes."""

    channels = 4

    def __init__(
        self,
        mel_frequencies: Tensor,
        *,
        sample_rate: int,
        f0_min: float,
        f0_max: float,
        narrow_bandwidth_semitones: float,
        wide_bandwidth_semitones: float,
        nyquist_ratio: float = 0.95,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
    ) -> None:
        super().__init__()
        if mel_frequencies.ndim != 1 or (mel_frequencies <= 0).any():
            raise ValueError("mel frequencies must be a positive vector")
        self.sample_rate = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.narrow_bandwidth = narrow_bandwidth_semitones
        self.wide_bandwidth = wide_bandwidth_semitones
        self.nyquist_ratio = nyquist_ratio
        upper = min(float(mel_frequencies.max()), sample_rate * 0.5 * nyquist_ratio)
        self.max_harmonic = max(1, math.ceil(upper / f0_min))
        self.register_buffer("mel_frequencies", mel_frequencies.float())
        mean = torch.zeros(self.channels) if feature_mean is None else feature_mean
        std = torch.ones(self.channels) if feature_std is None else feature_std
        if mean.shape != (self.channels,) or std.shape != (self.channels,):
            raise ValueError("harmonic normalization must have four channels")
        if (std <= 0).any():
            raise ValueError("harmonic feature std must be positive")
        self.register_buffer("feature_mean", mean.float())
        self.register_buffer("feature_std", std.float())

    def forward(self, f0: Tensor, voiced: Tensor | None = None) -> Tensor:
        if f0.shape[-1:] != (1,):
            raise ValueError("f0 must have shape [..., 1]")
        f0_work = f0.float()
        explicit_voiced = f0_work > 0 if voiced is None else voiced.bool()
        valid = (
            explicit_voiced
            & torch.isfinite(f0_work)
            & (f0_work >= self.f0_min)
            & (f0_work <= self.f0_max)
        )
        safe_f0 = torch.where(valid, f0_work, torch.ones_like(f0_work))
        scalar_f0 = safe_f0.squeeze(-1)
        frequencies = self.mel_frequencies.float()
        ratio = frequencies.view(*([1] * (f0.ndim - 1)), -1) / safe_f0
        nearest_index = ratio.round().clamp(1, self.max_harmonic)
        nearest_frequency = nearest_index * safe_f0
        distance = 12.0 * torch.log2(
            frequencies.view(*([1] * (f0.ndim - 1)), -1) / nearest_frequency
        )
        signed_distance = distance.clamp(-6, 6) / 6

        upper = min(
            float(self.mel_frequencies.max()),
            self.sample_rate * 0.5 * self.nyquist_ratio,
        )
        narrow, wide = self._occupancy(scalar_f0, frequencies, upper)
        narrow = narrow / narrow.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        wide = wide / wide.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        harmonic_index = torch.log1p(nearest_index) / math.log1p(self.max_harmonic)
        features = torch.stack((narrow, wide, signed_distance, harmonic_index), dim=-2)
        normalization_shape = [1] * (features.ndim - 2) + [self.channels, 1]
        features = (
            features - self.feature_mean.view(normalization_shape)
        ) / self.feature_std.view(normalization_shape)
        # Standardization must not turn unvoiced frames into nonzero features.
        return features * valid.to(features.dtype).unsqueeze(-1)

    def _occupancy(
        self, f0: Tensor, frequencies: Tensor, upper: float
    ) -> tuple[Tensor, Tensor]:
        flat_f0 = f0.reshape(-1)
        narrow_parts = []
        wide_parts = []
        # Bound temporary storage independently of batch and crop length.
        for frame_start in range(0, flat_f0.numel(), 1024):
            frame_f0 = flat_f0[frame_start : frame_start + 1024]
            narrow = torch.zeros(
                frame_f0.shape[0], frequencies.numel(), device=f0.device
            )
            wide = torch.zeros_like(narrow)
            for harmonic_start in range(1, self.max_harmonic + 1, 32):
                indices = torch.arange(
                    harmonic_start,
                    min(harmonic_start + 32, self.max_harmonic + 1),
                    device=f0.device,
                    dtype=torch.float32,
                )
                harmonic_frequencies = frame_f0[:, None] * indices[None]
                valid = harmonic_frequencies < upper
                distance = 12.0 * torch.log2(
                    frequencies[None, None, :]
                    / harmonic_frequencies[:, :, None].clamp_min(1e-12)
                )
                weights = indices.rsqrt()[None, :, None] * valid[:, :, None]
                narrow += (
                    torch.exp(-0.5 * (distance / self.narrow_bandwidth).square())
                    * weights
                ).sum(dim=1)
                wide += (
                    torch.exp(-0.5 * (distance / self.wide_bandwidth).square())
                    * weights
                ).sum(dim=1)
            narrow_parts.append(narrow)
            wide_parts.append(wide)
        shape = (*f0.shape, frequencies.numel())
        return torch.cat(narrow_parts).view(shape), torch.cat(wide_parts).view(shape)


def fit_harmonic_normalization(
    features: Tensor, voiced: Tensor
) -> tuple[Tensor, Tensor]:
    if features.shape[-2] != HarmonicFeatures.channels:
        raise ValueError("features must have four harmonic channels")
    selected = features.movedim(-2, -1)[voiced.bool().squeeze(-1)]
    if selected.numel() == 0:
        raise ValueError("cannot fit harmonic normalization without voiced frames")
    flattened = selected.reshape(-1, HarmonicFeatures.channels)
    return flattened.mean(dim=0), flattened.std(dim=0).clamp_min(1e-6)
