import pytest
import torch

from harp.f0 import (
    F0Track,
    FCPEEvidence,
    FCPEFrontend,
    decode_fcpe_local_average,
)


def test_f0_track_requires_canonical_shape_and_explicit_voicing() -> None:
    track = F0Track(
        f0_hz=torch.tensor([110.0, 0.0, 220.0]),
        voiced=torch.tensor([True, False, True]),
        native_score=torch.tensor([0.9, 0.1, 0.8]),
        sample_rate=48_000,
        hop_length=160,
        source_samples=480,
    )
    track.validate()
    with pytest.raises(ValueError, match="exactly equivalent"):
        F0Track(
            **{**vars(track), "voiced": torch.tensor([True, True, True])}
        ).validate()


def test_fcpe_decoder_returns_peak_score_and_local_weighted_pitch() -> None:
    cents = torch.arange(12, dtype=torch.float32) * 100 + 1200
    salience = torch.zeros(1, 2, 12)
    salience[0, 0, 4:7] = torch.tensor([0.25, 1.0, 0.25])
    salience[0, 1, 8] = 0.005
    evidence = FCPEEvidence(salience, cents, 16_000, 160, 16_000, 320)
    f0, score = decode_fcpe_local_average(evidence, threshold=0.006)
    assert score.tolist() == pytest.approx([1.0, 0.005])
    assert f0[0].item() == pytest.approx(10 * 2 ** (1700 / 1200))
    assert f0[1].item() == 0


def test_harp_fcpe_compatibility_is_exact_for_bundled_model() -> None:
    pytest.importorskip("torchfcpe")
    frontend = FCPEFrontend.load("cpu")
    sample_rate = 44_100
    samples = sample_rate // 4
    time = torch.arange(samples) / sample_rate
    waveform = 0.2 * torch.sin(2 * torch.pi * 440 * time)
    frames = samples // 512
    legacy = frontend.model.infer(
        waveform[None, :, None],
        sr=sample_rate,
        decoder_mode="local_argmax",
        threshold=0.006,
        f0_min=40,
        f0_max=1600,
        interp_uv=False,
        output_interp_target_length=frames,
    )
    legacy = legacy.squeeze(0).squeeze(-1).float().cpu()
    compatible = frontend.extract_compatible(
        waveform,
        sample_rate,
        target_sample_rate=sample_rate,
        target_hop_length=512,
        threshold=0.006,
        f0_min=40,
        f0_max=1600,
    )
    assert torch.equal(compatible.f0_hz, legacy)
