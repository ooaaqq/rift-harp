from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .flow_transform import FlowTransform
from .model import HARPCore


@dataclass(frozen=True)
class FlowCoefficients:
    c_in: Tensor
    c_skip: Tensor
    c_out: Tensor
    lambda_floor_fraction: float
    q_floor_fraction: float


@dataclass
class FlowLoss:
    total: Tensor
    flow: Tensor
    flow_by_sample: Tensor
    lambda_floor_fraction: float
    q_floor_fraction: float


def flow_coefficients(
    timestep: Tensor,
    lambda_raw: Tensor,
    *,
    lambda_floor: float = 1e-4,
    q_floor: float = 1e-6,
) -> FlowCoefficients:
    timestep = timestep.float()
    raw = lambda_raw.float().to(timestep.device)
    effective_lambda = raw.clamp_min(lambda_floor)
    if raw.ndim != 1:
        raise ValueError("lambda must be a mode vector")
    shape = (timestep.shape[0], 1, raw.shape[0])
    expanded_t = timestep[:, None, None]
    expanded_lambda = effective_lambda[None, None, :]
    q_raw = expanded_t.square() * expanded_lambda + (1 - expanded_t).square()
    q = q_raw.clamp_min(q_floor)
    coefficients = FlowCoefficients(
        c_in=q.rsqrt(),
        c_skip=(expanded_t * expanded_lambda - (1 - expanded_t)) / q,
        c_out=(expanded_lambda / q).sqrt(),
        lambda_floor_fraction=float((raw < lambda_floor).float().mean()),
        q_floor_fraction=float((q_raw < q_floor).float().mean()),
    )
    if coefficients.c_in.shape != shape:
        raise RuntimeError("unexpected flow coefficient shape")
    return coefficients


class HARPFlow(nn.Module):
    def __init__(
        self,
        model: HARPCore,
        transform: FlowTransform,
        *,
        speaker_drop_probability: float = 0.05,
        lambda_floor: float = 1e-4,
        q_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        self.model = model
        self.transform = transform
        self.speaker_drop_probability = speaker_drop_probability
        self.lambda_floor = lambda_floor
        self.q_floor = q_floor

    def forward(self, batch: dict[str, Tensor]) -> FlowLoss:
        raw_mel = batch["mel"].float()
        target = self.transform.transform(raw_mel)
        mask = batch["mask"]
        timestep = sample_timestep(target.shape[0], target.device)
        noise = torch.randn(target.shape, device=target.device, dtype=torch.float32)
        expanded_t = timestep[:, None, None]
        state = (1 - expanded_t) * noise + expanded_t * target
        target_velocity = target - noise
        coefficients = flow_coefficients(
            timestep,
            self.transform.lambda_raw.to(target.device),
            lambda_floor=self.lambda_floor,
            q_floor=self.q_floor,
        )
        residual_target = (
            target_velocity - coefficients.c_skip * state
        ) / coefficients.c_out
        speaker = batch["speaker"].clone()
        if self.training and self.speaker_drop_probability:
            dropped = (
                torch.rand(speaker.shape, device=speaker.device)
                < self.speaker_drop_probability
            )
            speaker[dropped] = self.model.null_speaker_id
        harmonic = batch.get("harmonic")
        if harmonic is None:
            harmonic = self.model.prepare_harmonic(batch["f0"])
        residual = self._model_residual(
            coefficients.c_in * state,
            batch["content"],
            batch["f0"],
            batch["rms"],
            harmonic,
            speaker,
            timestep,
            mask,
        )
        weights = mask.unsqueeze(-1).float()
        squared = (residual - residual_target).square() * weights
        denominator = weights.sum(dim=(1, 2)).clamp_min(1) * target.shape[-1]
        by_sample = squared.sum(dim=(1, 2)) / denominator
        flow = squared.sum() / denominator.sum()
        return FlowLoss(
            total=flow,
            flow=flow,
            flow_by_sample=by_sample,
            lambda_floor_fraction=coefficients.lambda_floor_fraction,
            q_floor_fraction=coefficients.q_floor_fraction,
        )

    @torch.inference_mode()
    def sample(
        self,
        content: Tensor,
        f0: Tensor,
        rms: Tensor,
        speaker: Tensor,
        mask: Tensor,
        *,
        steps: int = 32,
        guidance_strength: float = 1.0,
        method: str = "heun",
        generator: torch.Generator | None = None,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        if steps <= 0 or method not in {"euler", "heun"}:
            raise ValueError("invalid ODE sampler configuration")
        shape = (content.shape[0], content.shape[1], self.transform.channels)
        if initial_noise is None:
            state = torch.randn(
                shape,
                device=content.device,
                dtype=torch.float32,
                generator=generator,
            )
        else:
            if initial_noise.shape != shape:
                raise ValueError("initial noise shape does not match conditioning")
            state = initial_noise.float().to(content.device)
        state = state * mask.unsqueeze(-1).float()
        harmonic = self.model.prepare_harmonic(f0)
        times = torch.linspace(
            0, 1, steps + 1, device=state.device, dtype=torch.float32
        )
        for index in range(steps):
            timestep = times[index].expand(shape[0])
            delta = times[index + 1] - times[index]
            velocity = self._guided_velocity(
                state,
                content,
                f0,
                rms,
                harmonic,
                speaker,
                timestep,
                mask,
                guidance_strength,
            )
            proposal = state + delta * velocity
            if method == "heun" and index + 1 < steps:
                next_time = times[index + 1].expand(shape[0])
                next_velocity = self._guided_velocity(
                    proposal,
                    content,
                    f0,
                    rms,
                    harmonic,
                    speaker,
                    next_time,
                    mask,
                    guidance_strength,
                )
                state = state + delta * 0.5 * (velocity + next_velocity)
            else:
                state = proposal
            state = state * mask.unsqueeze(-1).float()
        return self.transform.inverse(state) * mask.unsqueeze(-1).float()

    def _guided_velocity(
        self,
        state: Tensor,
        content: Tensor,
        f0: Tensor,
        rms: Tensor,
        harmonic: Tensor,
        speaker: Tensor,
        timestep: Tensor,
        mask: Tensor,
        strength: float,
    ) -> Tensor:
        coefficients = flow_coefficients(
            timestep,
            self.transform.lambda_raw.to(state.device),
            lambda_floor=self.lambda_floor,
            q_floor=self.q_floor,
        )
        scaled_state = coefficients.c_in * state
        conditional = self._model_residual(
            scaled_state, content, f0, rms, harmonic, speaker, timestep, mask
        )
        if strength == 1.0:
            residual = conditional
        else:
            null_speaker = torch.full_like(speaker, self.model.null_speaker_id)
            unconditional = self._model_residual(
                scaled_state,
                content,
                f0,
                rms,
                harmonic,
                null_speaker,
                timestep,
                mask,
            )
            residual = unconditional + strength * (conditional - unconditional)
        return coefficients.c_skip * state + coefficients.c_out * residual

    def _model_residual(self, *args: Tensor) -> Tensor:
        dtype = next(self.model.parameters()).dtype
        inputs = [
            value.to(dtype) if value.is_floating_point() else value for value in args
        ]
        return self.model(*inputs).float()


def sample_timestep(batch: int, device: torch.device) -> Tensor:
    quantiles = torch.arange(batch, device=device, dtype=torch.float32) / batch
    uniform = quantiles + torch.rand(batch, device=device, dtype=torch.float32) / batch
    epsilon = torch.finfo(torch.float32).eps
    normal = torch.erfinv(uniform.clamp(epsilon, 1 - epsilon).mul(2).sub(1))
    values = torch.sigmoid(normal.mul(math.sqrt(2.0)))
    return values[torch.randperm(batch, device=device)]
