import torch

from harp.harmonic import HarmonicFeatures, mel_center_frequencies


def _extractor() -> HarmonicFeatures:
    return HarmonicFeatures(
        mel_center_frequencies(32, 40, 8000),
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
        mel_center_frequencies(16, 40, 4000),
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
