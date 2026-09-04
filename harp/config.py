from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    mel_channels: int = 128
    content_dim: int = 768
    dim: int = 1024
    depth: int = 16
    head_dim: int = 64
    ff_hidden_dim: int = 2816
    kernel_size: int = 31
    time_code_dim: int = 256
    speaker_code_dim: int = 256
    adaln_rank: int = 128
    adaln_mixer_dim: int = 256
    harmonic_dim: int = 128
    harmonic_injection_blocks: tuple[int, ...] = (4, 8, 12)
    activation_recompute_policy: str = "selective_expensive_ops"
    heavy_linear_precision: str = "float8_rowwise"


@dataclass(frozen=True)
class FlowConfig:
    transform_path: str = "artifacts/flow_transform_v1.npz"
    audit_path: str = "artifacts/flow_transform_v1.audit.json"
    lambda_floor: float = 1e-4
    q_floor: float = 1e-6
    timestep_sampling: str = "stratified_logit_normal"
    coefficient_dtype: str = "float32"
    ode_state_dtype: str = "float32"
    cfg_domain: str = "residual"


@dataclass(frozen=True)
class FeatureConfig:
    contract_path: str = "artifacts/feature_contract_v1.npz"
    sample_rate: int = 44_100
    hop_length: int = 512
    n_fft: int = 2048
    win_length: int = 2048
    power: float = 1.0
    center: bool = False
    pad_mode: str = "reflect"
    log_base: str = "natural"
    log_clamp: float = 1e-9
    mel_scale: str = "slaney"
    mel_norm: str = "slaney"
    rms_floor: float = 1e-5
    waveform_amplitude_convention: str = "float_-1_to_1"
    rms_definition: str = "waveform_frame_rms"


@dataclass(frozen=True)
class HarmonicConfig:
    sample_rate: int = 44100
    fmin: float = 40.0
    fmax: float = 16000.0
    f0_min: float = 40.0
    f0_max: float = 1600.0
    narrow_bandwidth_semitones: float = 0.35
    wide_bandwidth_semitones: float = 1.0
    nyquist_ratio: float = 0.95


@dataclass(frozen=True)
class TrainingConfig:
    warmup_valid_frames: int = 163_072_000
    max_steps: int = 500_000
    grad_clip_norm: float = 1.0
    ema_reference_frames: float = 16_307.2
    ema_reference_decay: float = 0.9999
    speaker_drop_probability: float = 0.05
    frame_buckets: tuple[int, ...] = (256, 384, 512)
    bucket_probabilities: tuple[float, ...] = (0.2, 0.3, 0.5)
    voiced_crop_probability: float = 0.7
    precision: str = "bfloat16"
    compile_mode: str = "max-autotune"
    inductor_epilogue_fusion: bool = True
    inductor_shape_padding: bool = True
    sdpa_backend: str = "cudnn"
    allow_tf32: bool = True
    log_every_steps: int = 20
    telemetry_every_steps: int = 500
    audit_checkpoint_every_valid_frames: int = 32_614_400
    full_checkpoint_every_valid_frames: int = 81_536_000
    health_every_valid_frames: int = 16_000_000
    local_audit_every_valid_frames: int = 80_000_000
    endpoint_every_valid_frames: int = 163_072_000
    full_panel_every_valid_frames: int = 326_144_000


@dataclass(frozen=True)
class SamplingConfig:
    dataset_probabilities: dict[str, float] = field(default_factory=dict)
    speaker_duration_exponent: float = 0.5
    speaker_probability_floor_ratio: float = 0.5
    speaker_probability_ceiling_ratio: float = 2.0
    song_duration_exponent: float = 0.5
    song_probability_floor_ratio: float = 0.5
    song_probability_ceiling_ratio: float = 2.0
    dataset_families: dict[str, str] = field(default_factory=dict)
    family_probability_caps: dict[str, float] = field(default_factory=dict)
    synthetic_datasets: tuple[str, ...] = ()
    max_singleton_real_speaker_median_ratio: float = 3.0
    batch_size: int = 96
    batch_frame_budget: int = 24_576
    steps_per_epoch: int = 1000
    seed: int = 2026
    num_workers: int = 8
    prefetch_factor: int = 2
    persistent_workers: bool = True
    canonical_bucket_padding: bool = True


@dataclass(frozen=True)
class OptimizerRoleConfig:
    learning_rate: float
    weight_decay: float


def _optimizer_roles() -> dict[str, OptimizerRoleConfig]:
    return {
        "backbone": OptimizerRoleConfig(1.5e-4, 0.010),
        "ff_expansion": OptimizerRoleConfig(3.0e-4, 0.005),
        "ff_contraction": OptimizerRoleConfig(7.5e-5, 0.020),
        "stem": OptimizerRoleConfig(1.5e-4, 0.010),
        "adaln": OptimizerRoleConfig(1.5e-4, 0.010),
        "harmonic_encoder": OptimizerRoleConfig(1.5e-4, 0.010),
        "harmonic_adapter": OptimizerRoleConfig(1.5e-4, 0.010),
        "output": OptimizerRoleConfig(1.5e-4, 0.010),
        "speaker": OptimizerRoleConfig(2.0e-4, 0.0),
        "branch_mix": OptimizerRoleConfig(1.5e-4, 0.0),
    }


@dataclass(frozen=True)
class OptimizerConfig:
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    roles: dict[str, OptimizerRoleConfig] = field(default_factory=_optimizer_roles)


@dataclass(frozen=True)
class ContractConfig:
    model_family: str = "rift-harp"
    architecture_version: int = 1
    flow_contract_version: int = 1
    feature_contract_version: int = 1
    checkpoint_schema: int = 1
    optimizer_role_map_version: int = 1


@dataclass(frozen=True)
class HARPConfig:
    num_speakers: int
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    harmonic: HarmonicConfig = field(default_factory=HarmonicConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)

    def validate(self) -> None:
        if self.contract.model_family != "rift-harp":
            raise ValueError("HARP config requires model_family='rift-harp'")
        if self.contract.checkpoint_schema != 1:
            raise ValueError("unsupported HARP checkpoint schema")
        if self.num_speakers <= 0:
            raise ValueError("num_speakers must be positive")
        model_sizes = (
            self.model.mel_channels,
            self.model.content_dim,
            self.model.dim,
            self.model.depth,
            self.model.head_dim,
            self.model.ff_hidden_dim,
            self.model.kernel_size,
            self.model.time_code_dim,
            self.model.speaker_code_dim,
            self.model.adaln_rank,
            self.model.adaln_mixer_dim,
            self.model.harmonic_dim,
        )
        if any(value <= 0 for value in model_sizes):
            raise ValueError("model dimensions must be positive")
        if self.model.kernel_size % 2 != 1:
            raise ValueError("depthwise convolution kernel must be odd")
        if self.model.dim % self.model.head_dim:
            raise ValueError("model dim must be divisible by head_dim")
        if self.model.head_dim % 2:
            raise ValueError("head_dim must be even")
        if self.model.heavy_linear_precision != "float8_rowwise":
            raise ValueError("HARP v1 requires rowwise FP8 heavy Linear training")
        if self.model.activation_recompute_policy != "selective_expensive_ops":
            raise ValueError("HARP v1 requires selective activation recomputation")
        if any(
            not 1 <= block < self.model.depth
            for block in self.model.harmonic_injection_blocks
        ):
            raise ValueError("harmonic injection blocks must be inside the backbone")
        if len(set(self.model.harmonic_injection_blocks)) != len(
            self.model.harmonic_injection_blocks
        ):
            raise ValueError("harmonic injection blocks must be unique")
        if abs(sum(self.training.bucket_probabilities) - 1.0) > 1e-6:
            raise ValueError("bucket probabilities must sum to one")
        if any(value <= 0 for value in self.training.bucket_probabilities):
            raise ValueError("bucket probabilities must be positive")
        if len(self.training.frame_buckets) != len(self.training.bucket_probabilities):
            raise ValueError("bucket lengths and probabilities differ")
        if any(value <= 0 for value in self.training.frame_buckets) or len(
            set(self.training.frame_buckets)
        ) != len(self.training.frame_buckets):
            raise ValueError("frame buckets must be distinct and positive")
        if self.optimizer.betas != (0.9, 0.95):
            raise ValueError("HARP v1 freezes AdamW betas at (0.9, 0.95)")
        if self.training.precision != "bfloat16":
            raise ValueError("HARP v1 supports BF16 model compute only")
        if self.flow.cfg_domain != "residual":
            raise ValueError("CFG must operate in the residual domain")
        if self.flow.timestep_sampling != "stratified_logit_normal":
            raise ValueError("unsupported timestep sampler")
        if self.flow.coefficient_dtype != "float32":
            raise ValueError("flow coefficients must remain FP32")
        if self.flow.ode_state_dtype != "float32":
            raise ValueError("ODE state must remain FP32")
        if self.feature.mel_scale != "slaney" or self.feature.mel_norm != "slaney":
            raise ValueError("HARP requires the exact Slaney mel contract")
        if self.feature.sample_rate != self.harmonic.sample_rate:
            raise ValueError("feature and harmonic sample rates must match")
        if (
            self.feature.hop_length <= 0
            or self.feature.n_fft <= 0
            or self.feature.win_length <= 0
            or self.feature.win_length > self.feature.n_fft
        ):
            raise ValueError("invalid mel window geometry")
        if self.feature.power != 1.0 or self.feature.log_base != "natural":
            raise ValueError("HARP v1 requires magnitude natural-log mel")
        if self.feature.center or self.feature.pad_mode != "reflect":
            raise ValueError("HARP v1 requires reflected explicit padding")
        if self.feature.log_clamp <= 0:
            raise ValueError("mel log clamp must be positive")
        if not (
            0 < self.harmonic.fmin < self.harmonic.fmax <= self.harmonic.sample_rate / 2
        ):
            raise ValueError("invalid mel frequency range")
        if not 0 < self.harmonic.f0_min <= self.harmonic.f0_max:
            raise ValueError("invalid F0 range")
        if not 0 < self.harmonic.nyquist_ratio <= 1:
            raise ValueError("nyquist_ratio must be in (0, 1]")
        if (
            self.harmonic.narrow_bandwidth_semitones <= 0
            or self.harmonic.wide_bandwidth_semitones
            < self.harmonic.narrow_bandwidth_semitones
        ):
            raise ValueError("harmonic bandwidths must be positive and ordered")
        if self.feature.rms_floor <= 0:
            raise ValueError("RMS floor must be positive")
        if self.training.warmup_valid_frames <= 0:
            raise ValueError("warmup_valid_frames must be positive")
        if self.training.max_steps <= 0 or self.training.grad_clip_norm <= 0:
            raise ValueError("training steps and gradient clip must be positive")
        if not 0 <= self.training.speaker_drop_probability < 1:
            raise ValueError("speaker dropout must be in [0, 1)")
        if not 0 <= self.training.voiced_crop_probability <= 1:
            raise ValueError("voiced crop probability must be in [0, 1]")
        if not 0 < self.training.ema_reference_decay < 1:
            raise ValueError("EMA reference decay must be in (0, 1)")
        if self.training.ema_reference_frames <= 0:
            raise ValueError("EMA reference frames must be positive")
        if self.sampling.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.sampling.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if self.sampling.batch_frame_budget < max(self.training.frame_buckets):
            raise ValueError("batch frame budget cannot fit the largest bucket")
        canonical_batches = {
            frames: min(
                self.sampling.batch_size,
                self.sampling.batch_frame_budget // frames,
            )
            for frames in self.training.frame_buckets
        }
        if any(
            batch * frames != self.sampling.batch_frame_budget
            for frames, batch in canonical_batches.items()
        ):
            raise ValueError(
                "every canonical batch must exactly fill batch_frame_budget"
            )
        if not self.sampling.canonical_bucket_padding:
            raise ValueError("canonical bucket padding must remain enabled")
        if self.sampling.num_workers <= 0 or self.sampling.prefetch_factor <= 0:
            raise ValueError(
                "production data loading requires positive workers/prefetch"
            )
        if not self.sampling.persistent_workers:
            raise ValueError("production data loading requires persistent workers")
        if self.training.sdpa_backend != "cudnn":
            raise ValueError("HARP v1 requires cuDNN SDPA")
        if not (
            self.training.inductor_epilogue_fusion
            and self.training.inductor_shape_padding
        ):
            raise ValueError("HARP v1 requires Inductor fusion and shape padding")
        cadence_values = (
            self.training.audit_checkpoint_every_valid_frames,
            self.training.full_checkpoint_every_valid_frames,
            self.training.health_every_valid_frames,
            self.training.local_audit_every_valid_frames,
            self.training.endpoint_every_valid_frames,
            self.training.full_panel_every_valid_frames,
        )
        if any(value <= 0 for value in cadence_values):
            raise ValueError("frame-based cadences must be positive")
        if set(self.optimizer.roles) != set(_optimizer_roles()):
            raise ValueError(
                "optimizer role table is incomplete or contains unknown roles"
            )
        for name, role in self.optimizer.roles.items():
            if (
                role.learning_rate <= 0
                or role.weight_decay < 0
                or self.optimizer.eps <= 0
            ):
                raise ValueError(f"invalid optimizer settings for role {name}")
        if not isinstance(self.sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig")
        if not self.sampling.dataset_probabilities:
            raise ValueError("dataset probabilities must not be empty")
        if abs(sum(self.sampling.dataset_probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("dataset probabilities must sum to one")
        if any(value <= 0 for value in self.sampling.dataset_probabilities.values()):
            raise ValueError("dataset probabilities must be positive")
        datasets = set(self.sampling.dataset_probabilities)
        if not set(self.sampling.dataset_families) <= datasets:
            raise ValueError("dataset families refer to unknown datasets")
        if not set(self.sampling.synthetic_datasets) <= datasets:
            raise ValueError("synthetic datasets refer to unknown datasets")
        if any(
            not 0 < cap <= 1 for cap in self.sampling.family_probability_caps.values()
        ):
            raise ValueError("family probability caps must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HARPConfig:
        model_payload = dict(payload["model"])
        model_payload["harmonic_injection_blocks"] = tuple(
            model_payload["harmonic_injection_blocks"]
        )
        training_payload = dict(payload["training"])
        training_payload["frame_buckets"] = tuple(training_payload["frame_buckets"])
        training_payload["bucket_probabilities"] = tuple(
            training_payload["bucket_probabilities"]
        )
        optimizer_payload = dict(payload["optimizer"])
        optimizer_payload["betas"] = tuple(optimizer_payload["betas"])
        optimizer_payload["roles"] = {
            name: OptimizerRoleConfig(**value)
            for name, value in optimizer_payload["roles"].items()
        }
        config = cls(
            num_speakers=int(payload["num_speakers"]),
            model=ModelConfig(**model_payload),
            flow=FlowConfig(**payload["flow"]),
            feature=FeatureConfig(**payload["feature"]),
            harmonic=HarmonicConfig(**payload["harmonic"]),
            training=TrainingConfig(**training_payload),
            sampling=SamplingConfig(
                **{
                    **payload["sampling"],
                    "synthetic_datasets": tuple(
                        payload["sampling"]["synthetic_datasets"]
                    ),
                }
            ),
            optimizer=OptimizerConfig(**optimizer_payload),
            contract=ContractConfig(**payload["contract"]),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> HARPConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
