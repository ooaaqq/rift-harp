import torch

from harp.singer_conversion import _core_weights, _window_starts


def test_window_starts_cover_song_with_fixed_stride() -> None:
    assert _window_starts(1000, 384, 64) == [0, 320, 640, 960]


def test_neighboring_core_fades_are_complementary() -> None:
    left = _core_weights(384, 64, first=True, last=False)
    right = _core_weights(384, 64, first=False, last=True)
    assert torch.allclose(left[-64:] + right[:64], torch.ones(64), atol=1e-6)
