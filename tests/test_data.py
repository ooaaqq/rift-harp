from pathlib import Path

import pytest
import torch

from harp.config import HARPConfig, ModelConfig, SamplingConfig, TrainingConfig
from harp.data import (
    FeatureDataset,
    HierarchicalBatchSampler,
    SampleRequest,
    bounded_normalize,
    collate_features,
)
from harp.manifest import ManifestEntry


def test_bounded_normalize_respects_bounds_without_order_bias() -> None:
    first = bounded_normalize([8, 1, 1, 1], 0.125, 0.5)
    second = bounded_normalize([1, 8, 1, 1], 0.125, 0.5)
    assert first == pytest.approx([0.5, 1 / 6, 1 / 6, 1 / 6])
    assert second == pytest.approx([1 / 6, 0.5, 1 / 6, 1 / 6])
    assert sum(first) == pytest.approx(1.0)
    assert all(0.125 <= value <= 0.5 for value in first)


def _entry(tmp_path: Path, index: int, dataset: str, speaker: str) -> ManifestEntry:
    prefix = tmp_path / f"features-{index}"
    frames = 20 + index
    torch.save(torch.randn(frames, 8), f"{prefix}.mel.pt")
    torch.save(torch.randn(frames // 2, 16), f"{prefix}.content.pt")
    torch.save(torch.rand(frames), f"{prefix}.f0.pt")
    torch.save(torch.rand(frames), f"{prefix}.rms.pt")
    return ManifestEntry(
        id=str(index),
        dataset=dataset,
        speaker=speaker,
        song=f"song-{index}",
        feature_prefix=str(prefix),
        frames=frames,
        split="train",
        quality_status="accepted",
    )


def test_real_sampler_path_loads_raw_aligned_features(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 0, "A", "alice"),
        _entry(tmp_path, 1, "B", "bob"),
    ]
    config = HARPConfig(
        num_speakers=2,
        model=ModelConfig(mel_channels=8, content_dim=16),
        training=TrainingConfig(
            frame_buckets=(8,),
            bucket_probabilities=(1.0,),
            voiced_crop_probability=0.7,
        ),
        sampling=SamplingConfig(
            dataset_probabilities={"A": 0.25, "B": 0.75},
            batch_size=2,
            batch_frame_budget=16,
            steps_per_epoch=2,
        ),
    )
    sampler = HierarchicalBatchSampler(entries, config)
    dataset = FeatureDataset(entries, 8, 16)
    requests = next(iter(sampler))
    batch = collate_features([dataset[request] for request in requests])
    assert batch["mel"].shape == (2, 8, 8)
    assert batch["content"].shape == (2, 8, 16)
    assert batch["mask"].all()
    assert sampler.dataset_probabilities == [0.25, 0.75]


def test_mel_only_stats_path_does_not_require_content(tmp_path: Path) -> None:
    entry = _entry(tmp_path, 0, "A", "alice")
    Path(f"{entry.feature_prefix}.content.pt").unlink()
    dataset = FeatureDataset([entry], 8, 16, mel_only=True)
    sample = dataset[0]
    assert "content" not in sample
    assert sample["mel"].shape == (20, 8)


def test_feature_length_mismatch_is_rejected_instead_of_truncated(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path, 0, "A", "alice")
    torch.save(torch.rand(entry.frames - 1), f"{entry.feature_prefix}.f0.pt")
    dataset = FeatureDataset([entry], 8, 16)
    with pytest.raises(ValueError, match="must match mel frames"):
        dataset[0]


def test_collate_always_pads_to_the_requested_bucket(tmp_path: Path) -> None:
    entry = _entry(tmp_path, 0, "A", "alice")
    dataset = FeatureDataset([entry], 8, 16)
    sample = dataset[SampleRequest(0, 32, 7)]
    batch = collate_features([sample])
    assert batch["mel"].shape == (1, 32, 8)
    assert int(batch["mask"].sum()) == 20
    assert not batch["mask"][0, 20:].any()


def test_sampler_resume_reproduces_uninterrupted_request_stream(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 0, "A", "alice"),
        _entry(tmp_path, 1, "B", "bob"),
    ]
    config = HARPConfig(
        num_speakers=2,
        model=ModelConfig(mel_channels=8, content_dim=16),
        training=TrainingConfig(
            frame_buckets=(8, 16),
            bucket_probabilities=(0.5, 0.5),
        ),
        sampling=SamplingConfig(
            dataset_probabilities={"A": 0.25, "B": 0.75},
            batch_size=2,
            batch_frame_budget=16,
            steps_per_epoch=8,
        ),
    )
    uninterrupted = HierarchicalBatchSampler(entries, config)
    all_batches = list(uninterrupted)

    resumed = HierarchicalBatchSampler(entries, config)
    resumed.set_epoch(0, 5)
    assert list(resumed) == all_batches[5:]
