import math

from harp.config import TrainingConfig
from harp.training_state import (
    TrainingProgress,
    cadence_crossed,
    ema_decay_for_batch,
    ema_half_life_valid_frames,
    warmup_scale,
)


def test_warmup_and_ema_are_defined_in_valid_frame_time() -> None:
    config = TrainingConfig()
    progress = TrainingProgress(seen_valid_frames=100_000_000)
    scale = warmup_scale(progress, 24_000_000, config.warmup_valid_frames)
    assert math.isclose(scale, 124_000_000 / 163_072_000)
    decay = ema_decay_for_batch(config, 24_576)
    assert math.isclose(decay, 0.9999 ** (24_576 / 16_307.2))
    expected_half_life = math.log(0.5) / math.log(0.9999) * 16_307.2
    assert math.isclose(ema_half_life_valid_frames(config), expected_half_life)


def test_frame_cadence_triggers_only_when_crossed() -> None:
    assert cadence_crossed(79_999_999, 80_000_001, 80_000_000)
    assert not cadence_crossed(80_000_001, 159_999_999, 80_000_000)
