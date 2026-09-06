import pytest
import torch

from harp.singer_conversion import (
    _core_weights,
    _filename_component,
    _resolve_foundation_speakers,
    _window_starts,
)


def test_window_starts_cover_song_with_fixed_stride() -> None:
    assert _window_starts(1000, 384, 64) == [0, 320, 640, 960]


def test_neighboring_core_fades_are_complementary() -> None:
    left = _core_weights(384, 64, first=True, last=False)
    right = _core_weights(384, 64, first=False, last=True)
    assert torch.allclose(left[-64:] + right[:64], torch.ones(64), atol=1e-6)


def test_resolve_foundation_speakers_preserves_request_order() -> None:
    assert _resolve_foundation_speakers(
        ["set:B", "set:A"], {"set:A": 2, "set:B": 7}
    ) == [
        ("set:B", 7),
        ("set:A", 2),
    ]


def test_resolve_foundation_speakers_rejects_unknown_and_duplicates() -> None:
    with pytest.raises(ValueError, match="unknown foundation speaker"):
        _resolve_foundation_speakers(["set:missing"], {"set:A": 2})
    with pytest.raises(ValueError, match="must be unique"):
        _resolve_foundation_speakers(["set:A", "set:A"], {"set:A": 2})


def test_filename_component_sanitizes_speaker_key() -> None:
    assert _filename_component("OpenSinger:female-35") == "OpenSinger-female-35"
