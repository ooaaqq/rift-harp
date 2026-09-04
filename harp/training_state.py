from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from torch import Tensor

from .config import TrainingConfig


@dataclass
class TrainingProgress:
    global_step: int = 0
    seen_requested_frames: int = 0
    seen_valid_frames: int = 0
    seen_crops: int = 0
    seen_voiced_frames: int = 0
    clipped_updates: int = 0
    seen_bucket_frames: dict[str, int] = field(
        default_factory=lambda: {"256": 0, "384": 0, "512": 0}
    )

    def batch_counts(self, batch: dict[str, Tensor]) -> dict[str, int]:
        valid = int(batch["length"].sum())
        requested = int(batch["requested_length"].sum())
        crops = int(batch["length"].numel())
        bucket_values = batch["requested_length"].unique()
        if bucket_values.numel() != 1:
            raise ValueError("training batches must contain one canonical bucket")
        bucket = int(bucket_values.item())
        mask = batch["mask"].bool()
        voiced = int(((batch["f0"][..., 0] > 0) & mask).sum())
        return {
            "valid_frames": valid,
            "requested_frames": requested,
            "crops": crops,
            "voiced_frames": voiced,
            "bucket": bucket,
        }

    def advance(self, counts: dict[str, int], *, clipped: bool = False) -> None:
        self.global_step += 1
        self.seen_valid_frames += counts["valid_frames"]
        self.seen_requested_frames += counts["requested_frames"]
        self.seen_crops += counts["crops"]
        self.seen_voiced_frames += counts["voiced_frames"]
        self.clipped_updates += int(clipped)
        key = str(counts["bucket"])
        self.seen_bucket_frames.setdefault(key, 0)
        self.seen_bucket_frames[key] += counts["valid_frames"]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TrainingProgress:
        return cls(
            global_step=int(payload["global_step"]),
            seen_requested_frames=int(payload["seen_requested_frames"]),
            seen_valid_frames=int(payload["seen_valid_frames"]),
            seen_crops=int(payload["seen_crops"]),
            seen_voiced_frames=int(payload["seen_voiced_frames"]),
            clipped_updates=int(payload["clipped_updates"]),
            seen_bucket_frames={
                str(name): int(value)
                for name, value in dict(payload["seen_bucket_frames"]).items()
            },
        )


def warmup_scale(
    progress: TrainingProgress, batch_valid_frames: int, target_frames: int
) -> float:
    return min(
        1.0,
        (progress.seen_valid_frames + batch_valid_frames) / target_frames,
    )


def ema_decay_for_batch(config: TrainingConfig, valid_frames: int) -> float:
    return config.ema_reference_decay ** (valid_frames / config.ema_reference_frames)


def ema_half_life_valid_frames(config: TrainingConfig) -> float:
    return (
        math.log(0.5)
        / math.log(config.ema_reference_decay)
        * config.ema_reference_frames
    )


def cadence_crossed(previous_frames: int, current_frames: int, interval: int) -> bool:
    if interval <= 0:
        raise ValueError("frame cadence must be positive")
    return current_frames // interval > previous_frames // interval
