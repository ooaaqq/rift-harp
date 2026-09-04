import torch

from harp.config import HarmonicConfig, ModelConfig
from harp.model import HARPCore
from harp.optimizer import (
    ROLE_HYPERPARAMETERS,
    build_optimizer,
    parameter_role_manifest,
)


def _small_model() -> HARPCore:
    return HARPCore(
        ModelConfig(
            mel_channels=8,
            content_dim=16,
            dim=32,
            depth=4,
            head_dim=8,
            ff_hidden_dim=64,
            kernel_size=5,
            time_code_dim=32,
            speaker_code_dim=32,
            adaln_rank=8,
            adaln_mixer_dim=16,
            harmonic_dim=8,
            harmonic_injection_blocks=(1, 2, 3),
        ),
        HarmonicConfig(sample_rate=16000, fmin=40, fmax=7600),
        num_speakers=3,
    )


def test_zero_initialized_core_is_a_zero_residual_predictor() -> None:
    model = _small_model()
    output = model(
        torch.randn(2, 9, 8),
        torch.randn(2, 9, 16),
        torch.rand(2, 9, 1) * 500 + 80,
        torch.randn(2, 9, 1),
        torch.tensor([0, 2]),
        torch.tensor([0.2, 0.8]),
        torch.tensor([[True] * 9, [True] * 7 + [False] * 2]),
    )
    assert output.count_nonzero() == 0


def test_low_rank_modulation_has_explicit_multiplicative_interaction() -> None:
    model = _small_model()
    modulation = model.blocks[0].modulation
    assert modulation.mixer.in_features == modulation.time_projection.out_features * 3
    assert modulation.output.weight.count_nonzero() == 0


def test_optimizer_roles_cover_every_trainable_parameter_once() -> None:
    model = _small_model()
    manifest = parameter_role_manifest(model)
    assert set(manifest) == {name for name, _ in model.named_parameters()}
    assert set(manifest.values()) == set(ROLE_HYPERPARAMETERS)
    optimizer = build_optimizer(model, fused=False)
    grouped = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters()
    }
