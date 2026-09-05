import pytest
import torch

from harp.speaker_audit_cli import normalized_progress, speaker_metrics


def test_speaker_metrics_use_target_minus_source_margin() -> None:
    metrics = speaker_metrics(
        torch.tensor([0.0, 2.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    )
    assert metrics == {
        "similarity_to_source": 0.0,
        "similarity_to_target": 1.0,
        "target_margin": 1.0,
    }


def test_normalized_progress_maps_source_and_target_anchors() -> None:
    assert normalized_progress(-0.4, -0.4, 0.6) == pytest.approx(0.0)
    assert normalized_progress(0.6, -0.4, 0.6) == pytest.approx(1.0)
    assert normalized_progress(0.1, -0.4, 0.6) == pytest.approx(0.5)


def test_normalized_progress_rejects_nonpositive_anchor() -> None:
    with pytest.raises(ValueError, match="positive anchor"):
        normalized_progress(0.0, 0.1, 0.1)
