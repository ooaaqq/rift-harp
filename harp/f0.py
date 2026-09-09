"""HARP-owned FCPE extraction and canonical F0 tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class F0Track:
    """A pitch track on one explicit, uniformly sampled frame grid."""

    f0_hz: Tensor
    voiced: Tensor
    native_score: Tensor | None
    sample_rate: int
    hop_length: int
    source_samples: int

    def validate(self) -> None:
        f0 = self.f0_hz.float().flatten()
        voiced = self.voiced.bool().flatten()
        expected = self.source_samples // self.hop_length
        if f0.shape != voiced.shape or f0.numel() != expected:
            raise ValueError("F0 and voiced must match the canonical frame count")
        if self.native_score is not None:
            score = self.native_score.float().flatten()
            if score.shape != f0.shape or not torch.isfinite(score).all():
                raise ValueError("native score must be finite and match F0")
        if not torch.isfinite(f0).all() or bool((f0 < 0).any()):
            raise ValueError("F0 must be finite and nonnegative")
        if not torch.equal(voiced, f0 > 0):
            raise ValueError("voiced must be exactly equivalent to positive F0")


@dataclass(frozen=True)
class FCPEEvidence:
    """Undecoded FCPE pitch-bin salience on the model-native frame grid."""

    salience: Tensor
    cent_table: Tensor
    native_sample_rate: int
    native_hop_length: int
    source_sample_rate: int
    source_samples: int


class FCPEFrontend:
    """Minimal FCPE model adapter with HARP-owned decoding and projection."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def load(cls, device: torch.device | str) -> FCPEFrontend:
        from torchfcpe import spawn_bundled_infer_model

        return cls(spawn_bundled_infer_model(device=str(device)))

    @property
    def native_sample_rate(self) -> int:
        return int(self.model.get_model_sr())

    @property
    def native_hop_length(self) -> int:
        return int(self.model.get_hop_size())

    @torch.inference_mode()
    def observe(
        self,
        waveform: Tensor,
        sample_rate: int,
    ) -> FCPEEvidence:
        batch = _waveform_batch(waveform).to(self.model.get_device())
        mel = self.model.wav2mel(batch, sample_rate)
        salience = self.model.model(mel)
        return FCPEEvidence(
            salience=salience,
            cent_table=self.model.model.cent_table,
            native_sample_rate=self.native_sample_rate,
            native_hop_length=self.native_hop_length,
            source_sample_rate=sample_rate,
            source_samples=batch.shape[1],
        )

    @torch.inference_mode()
    def extract_compatible(
        self,
        waveform: Tensor,
        sample_rate: int,
        *,
        target_sample_rate: int,
        target_hop_length: int,
        threshold: float,
        f0_min: float,
        f0_max: float,
    ) -> F0Track:
        """Reproduce legacy FCPE inference with HARP-owned postprocessing."""
        if sample_rate != target_sample_rate:
            raise ValueError("waveform must be resampled to the target sample rate")
        # Legacy FCPE only uses this value to derive UV when UV is requested.
        _ = f0_min
        source_samples = waveform.flatten().numel()
        target_frames = source_samples // target_hop_length
        evidence = self.observe(waveform, sample_rate)
        native_f0, score = decode_fcpe_local_average(evidence, threshold)
        projected = project_legacy_nearest(
            native_f0.clamp_max(f0_max), target_frames
        )
        projected_score = project_legacy_nearest(score, target_frames)
        track = F0Track(
            f0_hz=projected.cpu(),
            voiced=(projected > 0).cpu(),
            native_score=projected_score.cpu(),
            sample_rate=target_sample_rate,
            hop_length=target_hop_length,
            source_samples=source_samples,
        )
        track.validate()
        return track


def decode_fcpe_local_average(
    evidence: FCPEEvidence, threshold: float
) -> tuple[Tensor, Tensor]:
    """Decode the nine bins around each salience maximum, as legacy FCPE does."""
    salience = evidence.salience
    if salience.ndim != 3 or evidence.cent_table.ndim != 1:
        raise ValueError("FCPE evidence must have shapes [B,T,K] and [K]")
    score, peak = salience.max(dim=-1, keepdim=True)
    offsets = torch.arange(9, device=peak.device).view(1, 1, 9) - 4
    indices = (peak + offsets).clamp(0, salience.shape[-1] - 1)
    local_score = torch.gather(salience, -1, indices)
    cents = torch.gather(
        evidence.cent_table.view(1, 1, -1).expand_as(salience), -1, indices
    )
    decoded_cents = (cents * local_score).sum(-1) / local_score.sum(-1)
    f0 = 10.0 * torch.pow(2.0, decoded_cents / 1200.0)
    f0 = torch.where(score.squeeze(-1) > threshold, f0, torch.zeros_like(f0))
    return f0.squeeze(0), score.squeeze(0).squeeze(-1)


def project_legacy_nearest(values: Tensor, target_frames: int) -> Tensor:
    if target_frames <= 0:
        raise ValueError("target frame count must be positive")
    return F.interpolate(
        values.flatten()[None, None], size=target_frames, mode="nearest"
    )[0, 0]


def _waveform_batch(waveform: Tensor) -> Tensor:
    waveform = waveform.float()
    if waveform.ndim == 1:
        waveform = waveform[None, :, None]
    elif waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform[..., None]
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[-1] != 1:
        raise ValueError("FCPE frontend requires one mono waveform")
    return waveform
