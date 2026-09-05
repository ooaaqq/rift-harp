import pytest
import torch

from harp.common_panel_cli import (
    aggregate_samples,
    fixed_panel_noise,
    grouped_bootstrap_ci,
    paired_comparison,
    sample_metrics,
    waveform_dbfs,
)
from harp.flow_transform import orthonormal_dct


def test_fixed_panel_noise_uses_nested_length_prefixes() -> None:
    noise = fixed_panel_noise(3, 768, 128, 20260904)
    repeated = fixed_panel_noise(3, 768, 128, 20260904)
    assert torch.equal(noise, repeated)
    assert torch.equal(noise[:, :256], repeated[:, :256])


def test_sample_metrics_use_active_raw_mse_as_primary() -> None:
    target = torch.zeros(3, 128)
    prediction = torch.stack(
        [torch.ones(128), torch.full((128,), 2.0), torch.full((128,), 9.0)]
    )
    metrics = sample_metrics(
        prediction,
        target,
        torch.tensor([[0.1], [0.0], [0.1]]),
        torch.tensor([True, True, False]),
        orthonormal_dct(128),
    )
    assert metrics["active_frames"] == 1
    assert metrics["silence_frames"] == 1
    assert metrics["active_raw_mse"] == pytest.approx(1.0)
    assert metrics["silence_raw_mse"] == pytest.approx(4.0)
    assert metrics["full_raw_mse"] == pytest.approx(2.5)
    assert not metrics["catastrophe"]


def test_paired_comparison_pairs_by_ordinal_and_entry_id() -> None:
    panel = [
        {"ordinal": 0, "entry_id": "a", "song_key": "d:s1"},
        {"ordinal": 1, "entry_id": "b", "song_key": "d:s2"},
    ]
    before = [
        {"ordinal": 0, "entry_id": "a", "active_raw_mse": 2.0},
        {"ordinal": 1, "entry_id": "b", "active_raw_mse": 4.0},
    ]
    after = [
        {"ordinal": 1, "entry_id": "b", "active_raw_mse": 5.0},
        {"ordinal": 0, "entry_id": "a", "active_raw_mse": 1.0},
    ]
    result = paired_comparison(before, after, panel, bootstrap_samples=100, seed=1)
    assert result["mean_active_mse_gap"] == pytest.approx(0.0)
    assert result["harp_win_rate"] == pytest.approx(0.5)


def test_grouped_bootstrap_preserves_multi_crop_song_units() -> None:
    interval = grouped_bootstrap_ci(
        {"song-a": [-1.0, -1.0], "song-b": [1.0]},
        bootstrap_samples=1000,
        seed=7,
    )
    assert interval[0] == pytest.approx(-1.0)
    assert interval[1] == pytest.approx(1.0)


def test_aggregate_and_waveform_guardrails() -> None:
    rows = [
        {
            "active_raw_mse": 1.0,
            "full_raw_mse": 2.0,
            "silence_raw_mse": None,
            "catastrophe": False,
        },
        {
            "active_raw_mse": 3.0,
            "full_raw_mse": 6.0,
            "silence_raw_mse": 9.0,
            "catastrophe": True,
        },
    ]
    aggregate = aggregate_samples(rows)
    assert aggregate["mean_active_raw_mse"] == pytest.approx(2.0)
    assert aggregate["median_active_raw_mse"] == pytest.approx(2.0)
    assert aggregate["catastrophe_count"] == 1
    dbfs = waveform_dbfs(torch.ones(32))
    assert dbfs["rms_dbfs"] == pytest.approx(0.0)
    assert dbfs["peak_dbfs"] == pytest.approx(0.0)
