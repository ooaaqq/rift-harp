from pathlib import Path

import pytest

from harp.config import HARPConfig, SamplingConfig


def test_foundation_config_is_frozen_and_hashable() -> None:
    path = Path(__file__).parents[1] / "configs" / "foundation.json"
    config = HARPConfig.load(path)
    assert config.contract.model_family == "rift-harp"
    assert config.training.betas == (0.9, 0.95)
    assert config.model.harmonic_injection_blocks == (4, 8, 12)
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
