from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .flow_transform import fit_flow_transform


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit flow_transform_v1 from sampler-exposure batch files"
    )
    parser.add_argument("--batch-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sampler-config", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--crop-policy", required=True)
    parser.add_argument("--seed", type=int, default=44017)
    parser.add_argument("--max-frames", type=int, default=5_000_000)
    args = parser.parse_args()

    paths = [
        Path(line.strip())
        for line in Path(args.batch_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not paths:
        raise ValueError("batch list is empty")
    frame_parts = []
    frame_count = 0
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mel = payload["mel"].float()
        mask = payload["mask"].bool()
        selected = mel[mask]
        remaining = args.max_frames - frame_count
        if remaining <= 0:
            break
        frame_parts.append(selected[:remaining])
        frame_count += min(selected.shape[0], remaining)
    if frame_count < 100_000:
        raise ValueError("transform statistics require at least 100k valid frames")
    transform = fit_flow_transform(
        torch.cat(frame_parts),
        seed=args.seed,
        sampler_hash=_sha256(args.sampler_config),
        dataset_manifest_hash=_sha256(args.dataset_manifest),
        crop_policy_hash=_sha256(args.crop_policy),
    )
    digest = transform.save(args.output)
    summary = {
        "output": str(Path(args.output).resolve()),
        "sha256": digest,
        **transform.metadata,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
