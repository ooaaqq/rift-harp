import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from .config import HarmonicConfig, ModelConfig
from .feature_contract import FeatureContract
from .harmonic import HarmonicFeatures

_EXPENSIVE_OPS = (
    torch.ops.aten._scaled_mm.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.convolution.default,
    torch.ops.aten.silu.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
)


def _selective_checkpoint_contexts():
    def policy(_context, operation, *args, **kwargs):
        del args, kwargs
        if operation in _EXPENSIVE_OPS:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy)


class Attention(nn.Module):
    def __init__(self, dim: int, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.heads = dim // head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        batch, frames, dim = x.shape
        q, k, value = _linear_frames(self.qkv, _masked(x, mask)).chunk(3, dim=-1)
        q = self.q_norm(q.view(batch, frames, self.heads, self.head_dim)).transpose(
            1, 2
        )
        k = self.k_norm(k.view(batch, frames, self.heads, self.head_dim)).transpose(
            1, 2
        )
        value = value.view(batch, frames, self.heads, self.head_dim).transpose(1, 2)
        q, k = _rotary(q, k)
        attention_mask = None if mask is None else mask[:, None, None, :]
        result = F.scaled_dot_product_attention(
            q,
            k,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scale,
        )
        return _linear_frames(
            self.output,
            _masked(result.transpose(1, 2).reshape(batch, frames, dim), mask),
        )


class ConvFeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, kernel_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(dim, hidden * 2)
        self.conv = nn.Conv1d(
            hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden
        )
        self.output = nn.Linear(hidden, dim)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        value, gate = _linear_frames(self.input, _masked(x, mask)).chunk(2, dim=-1)
        value = _masked(value, mask)
        gate = _masked(gate, mask)
        value = self.conv(value.transpose(1, 2)).transpose(1, 2)
        gate = F.silu(gate)
        return _masked(_linear_frames(self.output, (value * gate).contiguous()), mask)


class LowRankModulation(nn.Module):
    def __init__(
        self, code_dim: int, rank: int, mixer_dim: int, output_dim: int
    ) -> None:
        super().__init__()
        self.time_projection = nn.Linear(code_dim, rank)
        self.speaker_projection = nn.Linear(code_dim, rank)
        self.mixer = nn.Linear(rank * 3, mixer_dim)
        self.output = nn.Linear(mixer_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        time_code: Tensor,
        speaker_code: Tensor,
    ) -> Tensor:
        time_low = self.time_projection(time_code)
        speaker_low = self.speaker_projection(speaker_code)
        mixed = torch.cat((time_low, speaker_low, time_low * speaker_low), dim=-1)
        return self.output(F.silu(self.mixer(mixed)))[:, None, :]


class AdaLNBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim, elementwise_affine=False)
        self.attention = Attention(config.dim, config.head_dim)
        self.norm2 = nn.LayerNorm(config.dim, elementwise_affine=False)
        self.feed_forward = ConvFeedForward(
            config.dim, config.ff_hidden_dim, config.kernel_size
        )
        self.modulation = LowRankModulation(
            config.time_code_dim,
            config.adaln_rank,
            config.adaln_mixer_dim,
            config.dim * 6,
        )

    def forward(
        self,
        x: Tensor,
        time_code: Tensor,
        speaker_code: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(
            time_code, speaker_code
        ).chunk(6, dim=-1)
        attended = self.attention(_modulate(self.norm1(x), shift_a, scale_a), mask)
        x = x + gate_a * attended
        fed = self.feed_forward(_modulate(self.norm2(x), shift_f, scale_f), mask)
        return _masked(x + gate_f * fed, mask)


class TimestepEmbedding(nn.Module):
    def __init__(self, output_dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, timestep: Tensor) -> Tensor:
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        phases = timestep.float()[:, None] * 1000.0 * frequencies[None]
        encoded = torch.cat((phases.cos(), phases.sin()), dim=-1)
        return self.mlp(encoded.to(self.mlp[0].weight.dtype))


class HARPCore(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        harmonic_config: HarmonicConfig,
        feature_contract: FeatureContract,
        num_speakers: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.null_speaker_id = num_speakers
        if feature_contract.channels != config.mel_channels:
            raise ValueError("feature contract mel channels differ from the model")
        self.harmonic_features = HarmonicFeatures(
            feature_contract.mel_center_hz,
            mel_fmax=harmonic_config.fmax,
            sample_rate=harmonic_config.sample_rate,
            f0_min=harmonic_config.f0_min,
            f0_max=harmonic_config.f0_max,
            narrow_bandwidth_semitones=harmonic_config.narrow_bandwidth_semitones,
            wide_bandwidth_semitones=harmonic_config.wide_bandwidth_semitones,
            nyquist_ratio=harmonic_config.nyquist_ratio,
            feature_mean=feature_contract.harmonic_mean,
            feature_std=feature_contract.harmonic_std,
        )
        self.rms_floor = feature_contract.rms_floor
        self.register_buffer(
            "rms_log_mean", torch.tensor(feature_contract.rms_log_mean).float()
        )
        self.register_buffer(
            "rms_log_std", torch.tensor(feature_contract.rms_log_std).float()
        )
        self.state_input = nn.Linear(config.mel_channels, config.dim)
        self.content_input = nn.Sequential(
            nn.Linear(config.content_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512, elementwise_affine=False),
        )
        self.pitch_input = nn.Sequential(
            nn.Linear(18, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128, elementwise_affine=False),
        )
        self.harmonic_input = nn.Sequential(
            nn.Linear(config.mel_channels * 4, 128),
            nn.SiLU(),
            nn.Linear(128, config.harmonic_dim),
            nn.LayerNorm(config.harmonic_dim, elementwise_affine=False),
        )
        self.energy_input = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64, elementwise_affine=False),
        )
        self.content_mix = nn.Parameter(torch.tensor(0.85))
        self.pitch_mix = nn.Parameter(torch.tensor(0.32))
        self.harmonic_mix = nn.Parameter(torch.tensor(0.32))
        self.energy_mix = nn.Parameter(torch.tensor(0.15))
        frame_dim = 512 + 128 + config.harmonic_dim + 64
        self.frame_condition = nn.Sequential(
            nn.Linear(frame_dim, config.dim),
            nn.LayerNorm(config.dim, elementwise_affine=False),
        )
        self.input_mix = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.LayerNorm(config.dim, elementwise_affine=False),
        )
        self.time = TimestepEmbedding(config.time_code_dim)
        self.speaker = nn.Embedding(num_speakers + 1, config.speaker_code_dim)
        self.blocks = nn.ModuleList([AdaLNBlock(config) for _ in range(config.depth)])
        self.harmonic_adapters = nn.ModuleDict(
            {
                str(block): nn.Linear(config.harmonic_dim, config.dim)
                for block in config.harmonic_injection_blocks
            }
        )
        self.final_norm = nn.LayerNorm(config.dim, elementwise_affine=False)
        self.final_modulation = LowRankModulation(
            config.time_code_dim,
            config.adaln_rank,
            config.adaln_mixer_dim,
            config.dim * 2,
        )
        self.output = nn.Linear(config.dim, config.mel_channels)
        self._semantic_initialize()

    def forward(
        self,
        scaled_state: Tensor,
        content: Tensor,
        f0: Tensor,
        rms: Tensor,
        harmonic_map: Tensor,
        speaker: Tensor,
        timestep: Tensor,
        mask: Tensor | None = None,
        speaker_code_override: Tensor | None = None,
    ) -> Tensor:
        voiced = torch.isfinite(f0) & (f0 > 0)
        harmonic = self.harmonic_input(harmonic_map.flatten(-2))
        pitch = self.pitch_input(_pitch_features(f0, voiced))
        frame = self.frame_condition(
            _magnitude_preserving_concat(
                (
                    self.content_input(content),
                    pitch,
                    harmonic,
                    self.energy_input(rms),
                ),
                (
                    self.content_mix,
                    self.pitch_mix,
                    self.harmonic_mix,
                    self.energy_mix,
                ),
            )
        )
        x = self.input_mix(torch.cat((self.state_input(scaled_state), frame), dim=-1))
        time_code = self.time(timestep)
        speaker_code = (
            self.speaker(speaker)
            if speaker_code_override is None
            else speaker_code_override
        )
        if speaker_code.ndim == 1:
            speaker_code = speaker_code.unsqueeze(0).expand(speaker.shape[0], -1)
        for index, block in enumerate(self.blocks, start=1):
            if str(index) in self.harmonic_adapters:
                x = x + self.harmonic_adapters[str(index)](harmonic)
            if (
                self.config.activation_recompute_policy
                == "selective_semantic_boundaries"
                and self.training
            ):
                x = checkpoint(
                    block,
                    x,
                    time_code,
                    speaker_code,
                    mask,
                    use_reentrant=False,
                    context_fn=_selective_checkpoint_contexts,
                )
            else:
                x = block(
                    x,
                    time_code,
                    speaker_code,
                    mask,
                )
        shift, scale = self.final_modulation(time_code, speaker_code).chunk(2, dim=-1)
        return _masked(self.output(_modulate(self.final_norm(x), shift, scale)), mask)

    def prepare_harmonic(self, f0: Tensor) -> Tensor:
        voiced = torch.isfinite(f0) & (f0 > 0)
        return self.harmonic_features(f0, voiced)

    def prepare_rms(self, rms: Tensor) -> Tensor:
        log_rms = torch.log(rms.float().clamp_min(0) + self.rms_floor)
        return (log_rms - self.rms_log_mean) / self.rms_log_std

    def branch_mix_fractions(self) -> dict[str, float]:
        names = ("content", "pitch", "harmonic", "energy")
        values = torch.stack(
            (self.content_mix, self.pitch_mix, self.harmonic_mix, self.energy_mix)
        ).float()
        fractions = values.square() / values.square().sum().clamp_min(1e-12)
        return {
            name: float(value.detach())
            for name, value in zip(names, fractions, strict=True)
        }

    def parameter_roles(self) -> Iterator[tuple[str, nn.Parameter, str]]:
        no_decay_names = {
            "content_mix",
            "pitch_mix",
            "harmonic_mix",
            "energy_mix",
        }
        for name, parameter in self.named_parameters():
            if name.startswith("speaker."):
                role = "speaker"
            elif name in no_decay_names:
                role = "branch_mix"
            elif ".feed_forward.input." in name:
                role = "ff_expansion"
            elif ".feed_forward.output." in name:
                role = "ff_contraction"
            elif ".modulation." in name or name.startswith("final_modulation."):
                role = "adaln"
            elif name.startswith("harmonic_input."):
                role = "harmonic_encoder"
            elif name.startswith("harmonic_adapters."):
                role = "harmonic_adapter"
            elif name.startswith("output."):
                role = "output"
            elif any(
                name.startswith(prefix)
                for prefix in (
                    "state_input.",
                    "content_input.",
                    "pitch_input.",
                    "energy_input.",
                    "frame_condition.",
                    "input_mix.",
                )
            ):
                role = "stem"
            else:
                role = "backbone"
            yield name, parameter, role

    def _semantic_initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, std=1 / math.sqrt(module.kernel_size[0]))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(
            self.speaker.weight, std=1 / math.sqrt(self.config.speaker_code_dim)
        )
        nn.init.normal_(
            self.state_input.weight, std=1 / math.sqrt(self.config.mel_channels)
        )
        nn.init.normal_(
            self.frame_condition[0].weight,
            std=1 / math.sqrt(self.frame_condition[0].in_features),
        )
        nn.init.normal_(
            self.input_mix[0].weight,
            std=1 / math.sqrt(self.input_mix[0].in_features),
        )
        for block in self.blocks:
            _spectral_chunks(
                block.attention.qkv.weight, 3, self.config.dim, self.config.dim
            )
            _spectral_chunks(
                block.feed_forward.input.weight,
                2,
                self.config.dim,
                self.config.ff_hidden_dim,
            )
            _spectral_normal(
                block.feed_forward.output.weight,
                self.config.ff_hidden_dim,
                self.config.dim,
            )
            _zero_linear(block.modulation.output)
        for adapter in self.harmonic_adapters.values():
            _zero_linear(adapter)
        _zero_linear(self.final_modulation.output)
        _zero_linear(self.output)


def _pitch_features(f0: Tensor, voiced: Tensor) -> Tensor:
    log_f0 = torch.where(
        voiced, torch.log2(f0.float().clamp_min(1.0) / 440.0), torch.zeros_like(f0)
    )
    frequencies = torch.arange(1, 9, device=f0.device, dtype=torch.float32)
    phases = log_f0 * frequencies * math.pi
    return torch.cat((log_f0, voiced.float(), phases.sin(), phases.cos()), dim=-1)


def _magnitude_preserving_concat(
    branches: tuple[Tensor, ...], mixing: tuple[Tensor, ...]
) -> Tensor:
    if len(branches) != len(mixing) or not branches:
        raise ValueError(
            "branches and mixing parameters must have equal nonzero length"
        )
    total_width = sum(branch.shape[-1] for branch in branches)
    mixing_values = torch.stack(mixing).float()
    denominator = mixing_values.square().sum().add(1e-12).sqrt()
    scaled = []
    for branch, value in zip(branches, mixing_values, strict=True):
        factor = value / denominator * math.sqrt(total_width / branch.shape[-1])
        scaled.append(branch * factor.to(branch.dtype))
    return torch.cat(scaled, dim=-1)


def _spectral_chunks(weight: Tensor, chunks: int, fan_in: int, fan_out: int) -> None:
    for chunk in weight.chunk(chunks, dim=0):
        _spectral_normal(chunk, fan_in, fan_out)


def _spectral_normal(weight: Tensor, fan_in: int, fan_out: int) -> None:
    std = 1 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    nn.init.normal_(weight, std=std)


def _zero_linear(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _masked(x: Tensor, mask: Tensor | None) -> Tensor:
    return x if mask is None else x * mask.unsqueeze(-1).to(x.dtype)


def _linear_frames(linear: nn.Module, x: Tensor) -> Tensor:
    shape = x.shape
    return linear(x.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def _rotary(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    dimension = q.shape[-1]
    positions = torch.arange(q.shape[-2], device=q.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(0, dimension, 2, device=q.device, dtype=torch.float32)
        / dimension
    )
    angles = positions[:, None] * frequencies[None]
    cosine = angles.cos().to(q.dtype)[None, None]
    sine = angles.sin().to(q.dtype)[None, None]
    return _apply_rotary(q, cosine, sine), _apply_rotary(k, cosine, sine)


def _apply_rotary(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)
