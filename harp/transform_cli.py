from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import HARPConfig
from .data import FeatureDataset, HierarchicalBatchSampler, collate_features
from .flow_transform import fit_flow_transform
from .manifest import load_manifest, manifest_sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit flow_transform_v1 from the production sampler exposure"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3_000_000)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if not 3_000_000 <= args.frames <= 5_000_000:
        raise ValueError("the frozen transform contract requires 3M to 5M frames")

    config = HARPConfig.load(args.config)
    entries = load_manifest(args.manifest)
    training_entries = [
        entry
        for entry in entries
        if entry.split == "train" and entry.quality_status == "accepted"
    ]
    dataset = FeatureDataset(
        training_entries,
        config.model.mel_channels,
        config.model.content_dim,
        voiced_crop_probability=config.training.voiced_crop_probability,
        mel_only=True,
    )
    sampler = HierarchicalBatchSampler(training_entries, config)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_features,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    mel_parts = []
    voiced_parts = []
    count = 0
    for batch in loader:
        valid = batch["mask"]
        mel = batch["mel"][valid]
        voiced = batch["f0"][valid][:, 0] > 0
        remaining = args.frames - count
        mel_parts.append(mel[:remaining])
        voiced_parts.append(voiced[:remaining])
        count += min(remaining, mel.shape[0])
        if count >= args.frames:
            break
    if count != args.frames:
        raise RuntimeError(f"sampler yielded only {count} of {args.frames} frames")

    frames = torch.cat(mel_parts)
    voiced = torch.cat(voiced_parts)
    crop_contract = {
        "frame_buckets": config.training.frame_buckets,
        "bucket_probabilities": config.training.bucket_probabilities,
        "voiced_crop_probability": config.training.voiced_crop_probability,
    }
    transform = fit_flow_transform(
        frames,
        seed=config.sampling.seed,
        sampler_hash=_json_sha256(config.sampling),
        dataset_manifest_hash=manifest_sha256(args.manifest),
        crop_policy_hash=_json_sha256(crop_contract),
    )
    transformed = transform.transform(frames)
    transform.metadata["exposure"] = {
        "voiced_fraction": float(voiced.float().mean()),
        "active_variance": _variance_summary(transformed[voiced]),
        "unvoiced_variance": _variance_summary(transformed[~voiced]),
    }
    digest = transform.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                **transform.metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _variance_summary(frames: torch.Tensor) -> dict[str, float | int]:
    if frames.shape[0] < 2:
        return {"frames": int(frames.shape[0])}
    variance = frames.double().var(dim=0, correction=1)
    return {
        "frames": int(frames.shape[0]),
        "min": float(variance.min()),
        "median": float(variance.median()),
        "max": float(variance.max()),
    }


def _json_sha256(value: object) -> str:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main()
