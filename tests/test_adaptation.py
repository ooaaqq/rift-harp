from pathlib import Path

import torch
from torch import nn

from harp.adaptation import (
    PseudoPairedBatchSampler,
    PseudoVariant,
    build_finetune_optimizer,
    configure_finetune_parameters,
)
from harp.config import HARPConfig, ModelConfig, SamplingConfig, TrainingConfig
from harp.feature_contract import neutral_feature_contract
from harp.manifest import ManifestEntry
from harp.model import HARPCore


def _model() -> HARPCore:
    config = ModelConfig(
        mel_channels=4,
        content_dim=8,
        dim=16,
        depth=1,
        head_dim=4,
        ff_hidden_dim=24,
        kernel_size=3,
        time_code_dim=16,
        speaker_code_dim=16,
        adaln_rank=4,
        adaln_mixer_dim=8,
        harmonic_dim=4,
        harmonic_injection_blocks=(),
    )
    return HARPCore(
        config,
        HARPConfig(num_speakers=2).harmonic,
        neutral_feature_contract(4, 40, 16000),
        2,
    )


def _entry(index: int, song: str, frames: int) -> ManifestEntry:
    return ManifestEntry(
        id=f"recording-{index}",
        dataset="Target",
        speaker="target",
        song=song,
        feature_prefix=f"unused-{index}",
        frames=frames,
        split="train",
        quality_status="accepted",
    )


def test_finetune_parameter_and_optimizer_coverage() -> None:
    model = _model()
    trainable, frozen = configure_finetune_parameters(model)
    assert frozen == [
        "content_mix",
        "pitch_mix",
        "harmonic_mix",
        "energy_mix",
        "speaker.weight",
    ]
    assert "time.mlp.0.weight" in trainable
    assert "blocks.0.modulation.output.weight" in trainable
    code = nn.Parameter(torch.zeros(model.config.speaker_code_dim))
    optimizer = build_finetune_optimizer(
        model,
        code,
        model_learning_rate=2e-5,
        code_learning_rate=1e-4,
        fused=False,
    )
    grouped = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert {id(value) for value in grouped} == {
        id(value) for value in expected + [code]
    }
    code_group = next(
        group for group in optimizer.param_groups if group["role"] == "target_code"
    )
    assert code_group["lr"] == 1e-4 and code_group["weight_decay"] == 0


def test_pseudo_sampler_keeps_target_marginal_independent_of_variant_count() -> None:
    entries = [_entry(0, "song-a", 100), _entry(1, "song-b", 100)]

    def variant(suffix: str) -> PseudoVariant:
        return PseudoVariant(suffix, str(Path(f"{suffix}.pt")), suffix, "accepted")

    variants = {
        "recording-0": [variant("b"), variant("c"), variant("d")],
        "recording-1": [variant("b")],
    }
    config = HARPConfig(
        num_speakers=2,
        training=TrainingConfig(frame_buckets=(4,), bucket_probabilities=(1.0,)),
        sampling=SamplingConfig(
            dataset_probabilities={"Target": 1.0},
            steps_per_epoch=200,
        ),
    )
    sampler = PseudoPairedBatchSampler(
        entries,
        variants,
        config,
        seed=7,
        pseudo_probability=0.7,
        batch_frame_budget=8,
    )
    requests = [request for batch in sampler for request in batch]
    shares = [
        sum(request.index == index for request in requests) / len(requests)
        for index in range(2)
    ]
    assert all(abs(share - 0.5) < 0.08 for share in shares)
    pseudo_share = sum(request.is_pseudo for request in requests) / len(requests)
    assert abs(pseudo_share - 0.7) < 0.08
