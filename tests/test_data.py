from pathlib import Path

import torch

from harp.config import HARPConfig, ModelConfig, SamplingConfig, TrainingConfig
from harp.data import FeatureDataset, HierarchicalBatchSampler, collate_features
from harp.manifest import ManifestEntry


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
