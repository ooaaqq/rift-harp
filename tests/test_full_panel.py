import torch

from harp.full_panel_cli import (
    _reference_crop,
    _select_by_length,
    pitch_metrics,
    tail_metrics,
)


def test_panel_selection_is_bounded_per_context_length() -> None:
    items = [
        {"requested_frames": length, "ordinal": ordinal}
        for ordinal in range(4)
        for length in (256, 512, 768)
    ]
    selected = _select_by_length(items, 2)
    assert len(selected) == 6
    assert {
        length: sum(item["requested_frames"] == length for item in selected)
        for length in (256, 512, 768)
    } == {256: 2, 512: 2, 768: 2}


def test_pitch_metrics_report_exact_tracking() -> None:
    target = torch.tensor([0.0, 110.0, 220.0, 0.0])
    assert pitch_metrics(target, target) == {
        "frames": 4,
        "voicing_precision": 1.0,
        "voicing_recall": 1.0,
        "voicing_f1": 1.0,
        "f0_cents_mae": 0.0,
        "gross_pitch_error_ratio": 0.0,
    }


def test_tail_metrics_detect_length_and_nonfinite_failures() -> None:
    waveform = torch.tensor([0.0, 1.0, 0.0, float("nan")])
    metrics = tail_metrics(waveform, expected_samples=5, hop_length=2)
    assert metrics["length_error"] == -1
    assert not metrics["finite"]


def test_reference_crop_resamples_before_frame_indexing(tmp_path) -> None:
    import soundfile as sf

    source_rate = 48_000
    path = tmp_path / "source.wav"
    sf.write(path, torch.zeros(source_rate).numpy(), source_rate)
    crop, observed_rate = _reference_crop(
        path,
        start_frame=2,
        frames=4,
        target_sample_rate=44_100,
        hop_length=512,
    )
    assert observed_rate == source_rate
    assert crop.shape == (4 * 512,)
