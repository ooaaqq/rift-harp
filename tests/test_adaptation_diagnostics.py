import pytest
import torch

from harp.adaptation_diagnostics import code_movement


def test_code_movement_reports_distances_and_cosine() -> None:
    result = code_movement(
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    )

    assert result["distance_from_initial"] == pytest.approx(2**0.5)
    assert result["distance_from_previous"] == pytest.approx(1.0)
    assert result["cosine_with_previous"] == pytest.approx(2**-0.5)
