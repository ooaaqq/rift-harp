import torch

from harp.feature_contract import slaney_mel_centers
from harp.harmonic import HarmonicFeatures


def _extractor() -> HarmonicFeatures:
    return HarmonicFeatures(
        slaney_mel_centers(32, 40, 8000),
        mel_fmax=8000,
        sample_rate=22050,
        f0_min=40,
        f0_max=1200,
        narrow_bandwidth_semitones=0.35,
        wide_bandwidth_semitones=1.0,
    )


def test_harmonic_features_handle_voicing_and_invalid_f0() -> None:
    extractor = _extractor()
    f0 = torch.tensor([[[110.0], [880.0], [0.0], [float("nan")], [4000.0]]])
    features = extractor(f0)
    assert features.shape == (1, 5, 4, 32)
    assert torch.isfinite(features).all()
    assert features[:, 2:].count_nonzero() == 0
    torch.testing.assert_close(features[0, :2, 0].amax(dim=-1), torch.ones(2))
    torch.testing.assert_close(features[0, :2, 1].amax(dim=-1), torch.ones(2))


def test_standardization_does_not_unmask_unvoiced_frames() -> None:
    extractor = HarmonicFeatures(
        slaney_mel_centers(16, 40, 4000),
        mel_fmax=4000,
        sample_rate=16000,
        f0_min=40,
        f0_max=1000,
        narrow_bandwidth_semitones=0.35,
        wide_bandwidth_semitones=1.0,
        feature_mean=torch.ones(4),
        feature_std=torch.full((4,), 2.0),
    )
    features = extractor(torch.zeros(2, 3, 1))
    assert features.count_nonzero() == 0


def test_slaney_centers_are_internal_filter_peaks() -> None:
    centers = slaney_mel_centers(128, 40, 16000)
    assert centers.shape == (128,)
    torch.testing.assert_close(centers[0], torch.tensor(68.28296), atol=1e-4, rtol=0)
    torch.testing.assert_close(centers[-1], torch.tensor(15540.0596), atol=2e-3, rtol=0)
    assert centers[0] > 40
    assert centers[-1] < 16000


def test_nearest_harmonic_is_clamped_per_frame() -> None:
    extractor = HarmonicFeatures(
        torch.tensor([3900.0]),
        mel_fmax=4000,
        sample_rate=16000,
        f0_min=40,
        f0_max=2000,
        narrow_bandwidth_semitones=0.35,
        wide_bandwidth_semitones=1.0,
    )
    features = extractor(torch.tensor([[[1500.0]]]))
    expected_index = torch.log1p(torch.tensor(2.0)) / torch.log1p(torch.tensor(100.0))
    torch.testing.assert_close(features[0, 0, 3, 0], expected_index)
