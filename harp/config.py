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


@dataclass(frozen=True)
class FlowConfig:
    transform_path: str = "artifacts/flow_transform_v1.npz"
    lambda_floor: float = 1e-4
    q_floor: float = 1e-6
    timestep_sampling: str = "stratified_logit_normal"
    coefficient_dtype: str = "float32"
    ode_state_dtype: str = "float32"
    cfg_domain: str = "residual"


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
    base_learning_rate: float = 1.5e-4
    speaker_learning_rate: float = 2.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 10_000
    max_steps: int = 500_000
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999
    speaker_drop_probability: float = 0.05
    frame_buckets: tuple[int, ...] = (256, 384, 512)
    bucket_probabilities: tuple[float, ...] = (0.2, 0.3, 0.5)
    voiced_crop_probability: float = 0.7
    precision: str = "bfloat16"
    compile_mode: str = "max-autotune"
    sdpa_backend: str = "cudnn"


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
    harmonic: HarmonicConfig = field(default_factory=HarmonicConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)

    def validate(self) -> None:
        if self.contract.model_family != "rift-harp":
            raise ValueError("HARP config requires model_family='rift-harp'")
        if self.contract.checkpoint_schema != 1:
            raise ValueError("unsupported HARP checkpoint schema")
        if self.num_speakers <= 0:
            raise ValueError("num_speakers must be positive")
        if self.model.dim % self.model.head_dim:
            raise ValueError("model dim must be divisible by head_dim")
        if self.model.head_dim % 2:
            raise ValueError("head_dim must be even")
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
        if len(self.training.frame_buckets) != len(self.training.bucket_probabilities):
            raise ValueError("bucket lengths and probabilities differ")
        if self.training.betas != (0.9, 0.95):
            raise ValueError("HARP v1 freezes AdamW betas at (0.9, 0.95)")
        if self.training.precision != "bfloat16":
            raise ValueError("HARP v1 supports BF16 model compute only")
        if self.flow.cfg_domain != "residual":
            raise ValueError("CFG must operate in the residual domain")

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
    def load(cls, path: str | Path) -> HARPConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model_payload = dict(payload["model"])
        model_payload["harmonic_injection_blocks"] = tuple(
            model_payload["harmonic_injection_blocks"]
        )
        training_payload = dict(payload["training"])
        training_payload["betas"] = tuple(training_payload["betas"])
        training_payload["frame_buckets"] = tuple(training_payload["frame_buckets"])
        training_payload["bucket_probabilities"] = tuple(
            training_payload["bucket_probabilities"]
        )
        config = cls(
            num_speakers=int(payload["num_speakers"]),
            model=ModelConfig(**model_payload),
            flow=FlowConfig(**payload["flow"]),
            harmonic=HarmonicConfig(**payload["harmonic"]),
            training=TrainingConfig(**training_payload),
            contract=ContractConfig(**payload["contract"]),
        )
        config.validate()
        return config
