import torch

from harp.config import HarmonicConfig, ModelConfig, OptimizerConfig
from harp.feature_contract import neutral_feature_contract
from harp.model import _EXPENSIVE_OPS, HARPCore, _magnitude_preserving_concat
from harp.optimizer import build_optimizer, parameter_role_manifest
from harp.precision import heavy_linear_names


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
        neutral_feature_contract(8, 40, 7600),
        num_speakers=3,
    )


def test_selective_recompute_saves_only_real_expensive_producers() -> None:
    assert set(_EXPENSIVE_OPS) == {
        torch.ops.aten._scaled_mm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.convolution.default,
        torch.ops.aten.silu.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    }


def test_zero_initialized_core_is_a_zero_residual_predictor() -> None:
    model = _small_model()
    f0 = torch.rand(2, 9, 1) * 500 + 80
    output = model(
        torch.randn(2, 9, 8),
        torch.randn(2, 9, 16),
        f0,
        torch.randn(2, 9, 1),
        model.prepare_harmonic(f0),
        torch.tensor([0, 0]),
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
    assert set(manifest.values()) == set(OptimizerConfig().roles)
    optimizer = build_optimizer(model, OptimizerConfig(), fused=False)
    grouped = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters()
    }


def test_only_four_heavy_linears_per_block_are_selected_for_fp8() -> None:
    model = _small_model()
    assert heavy_linear_names(model) == tuple(
        f"blocks.{block}.{suffix}"
        for block in range(model.config.depth)
        for suffix in (
            "attention.qkv",
            "attention.output",
            "feed_forward.input",
            "feed_forward.output",
        )
    )


def test_magnitude_preserving_concat_uses_mix_fractions_not_widths() -> None:
    branches = (
        torch.randn(4096, 512),
        torch.randn(4096, 128),
        torch.randn(4096, 128),
        torch.randn(4096, 64),
    )
    mixing = tuple(torch.tensor(value) for value in (0.85, 0.32, 0.32, 0.15))
    combined = _magnitude_preserving_concat(branches, mixing)
    measured = []
    start = 0
    for branch in branches:
        end = start + branch.shape[-1]
        measured.append(float(combined[:, start:end].square().sum()))
        start = end
    measured = torch.tensor(measured) / sum(measured)
    expected = torch.tensor([0.85, 0.32, 0.32, 0.15]).square()
    expected /= expected.sum()
    torch.testing.assert_close(measured, expected, atol=0.01, rtol=0)


def test_state_projection_has_unit_scale_semantic_initialization() -> None:
    model = _small_model()
    values = torch.randn(20_000, 8)
    output_rms = model.state_input(values).square().mean().sqrt()
    torch.testing.assert_close(output_rms, torch.tensor(1.0), atol=0.08, rtol=0)


def test_speaker_code_override_matches_direct_speaker_route() -> None:
    model = _small_model()
    f0 = torch.rand(2, 9, 1) * 500 + 80
    inputs = (
        torch.randn(2, 9, 8),
        torch.randn(2, 9, 16),
        f0,
        torch.randn(2, 9, 1),
        model.prepare_harmonic(f0),
        torch.tensor([0, 0]),
        torch.tensor([0.2, 0.8]),
        torch.ones(2, 9, dtype=torch.bool),
    )
    model.output.weight.data.normal_()
    for block in model.blocks:
        block.modulation.output.weight.data.normal_()
    model.final_modulation.output.weight.data.normal_()
    baseline = model(*inputs)
    overridden = model(
        *inputs,
        speaker_code_override=model.speaker.weight[0].detach(),
    )
    torch.testing.assert_close(overridden, baseline)


def test_selective_recompute_matches_eager_forward_and_gradients() -> None:
    checkpointed = _small_model()
    eager_config = ModelConfig(
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
        activation_recompute_policy="disabled_for_equivalence_test",
    )
    eager = HARPCore(
        eager_config,
        HarmonicConfig(sample_rate=16000, fmin=40, fmax=7600),
        neutral_feature_contract(8, 40, 7600),
        num_speakers=3,
    )
    eager.load_state_dict(checkpointed.state_dict())
    inputs = (
        torch.randn(2, 9, 8),
        torch.randn(2, 9, 16),
        torch.rand(2, 9, 1) * 500 + 80,
        torch.randn(2, 9, 1),
        torch.randn(2, 9, 8, 4),
        torch.tensor([0, 2]),
        torch.tensor([0.2, 0.8]),
        torch.tensor([[True] * 9, [True] * 7 + [False] * 2]),
    )
    checkpointed.output.weight.data.normal_()
    eager.output.weight.data.copy_(checkpointed.output.weight.data)
    checkpointed_output = checkpointed(*inputs)
    eager_output = eager(*inputs)
    torch.testing.assert_close(checkpointed_output, eager_output)
    checkpointed_output.square().mean().backward()
    eager_output.square().mean().backward()
    for checkpointed_parameter, eager_parameter in zip(
        checkpointed.parameters(), eager.parameters(), strict=True
    ):
        torch.testing.assert_close(
            checkpointed_parameter.grad, eager_parameter.grad, equal_nan=True
        )
