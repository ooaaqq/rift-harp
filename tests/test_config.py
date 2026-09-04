from pathlib import Path

import pytest

from harp.config import HARPConfig, SamplingConfig


def test_foundation_config_is_frozen_and_hashable() -> None:
    path = Path(__file__).parents[1] / "configs" / "foundation.json"
    config = HARPConfig.load(path)
    assert config.contract.model_family == "rift-harp"
    assert config.optimizer.betas == (0.9, 0.95)
    assert config.model.harmonic_injection_blocks == (4, 8, 12)
    assert config.model.activation_recompute_policy == "none"
    assert config.model.heavy_linear_precision == "float8_rowwise"
    assert config.training.compile_mode == "max-autotune"
    assert config.sampling.batch_size == 96
    assert config.sampling.batch_frame_budget == 24_576
    assert {
        frames: min(
            config.sampling.batch_size,
            config.sampling.batch_frame_budget // frames,
        )
        for frames in config.training.frame_buckets
    } == {256: 96, 384: 64, 512: 48}
    assert config.training.warmup_valid_frames == 163_072_000
    assert len(config.digest()) == 64


def test_old_checkpoint_family_is_rejected() -> None:
    from harp.checkpoint import validate_checkpoint_contract

    config = HARPConfig(
        num_speakers=3,
        sampling=SamplingConfig(dataset_probabilities={"test": 1.0}),
    )
    with pytest.raises(ValueError, match="not a RIFT-HARP"):
        validate_checkpoint_contract(
            {"model_family": "rift-svc-v4", "checkpoint_schema": 4}, config
        )
