from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import HARPConfig
from .contracts import exposure_semantics_hash, stats_run_hash
from .data import FeatureDataset, HierarchicalBatchSampler, collate_features
from .feature_contract import (
    FeatureStatisticsAccumulator,
    slaney_mel_centers,
)
from .harmonic import HarmonicFeatures
from .manifest import load_manifest, manifest_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit HARP feature_contract_v1")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3_000_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 3_000_000 <= args.frames <= 5_000_000:
        raise ValueError("the feature contract requires 3M to 5M valid frames")

    config = HARPConfig.load(args.config)
    entries = [
        entry
        for entry in load_manifest(args.manifest)
        if entry.split == "train" and entry.quality_status == "accepted"
    ]
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        voiced_crop_probability=config.training.voiced_crop_probability,
        mel_only=True,
    )
    sampler = HierarchicalBatchSampler(entries, config)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_features,
        num_workers=config.sampling.num_workers,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=config.sampling.prefetch_factor,
        persistent_workers=config.sampling.persistent_workers,
    )
    device = torch.device(args.device)
    centers = slaney_mel_centers(
        config.model.mel_channels, config.harmonic.fmin, config.harmonic.fmax
    ).to(device)
    raw_harmonic = HarmonicFeatures(
        centers,
        mel_fmax=config.harmonic.fmax,
        sample_rate=config.harmonic.sample_rate,
        f0_min=config.harmonic.f0_min,
        f0_max=config.harmonic.f0_max,
        narrow_bandwidth_semitones=config.harmonic.narrow_bandwidth_semitones,
        wide_bandwidth_semitones=config.harmonic.wide_bandwidth_semitones,
        nyquist_ratio=config.harmonic.nyquist_ratio,
    ).to(device)
    accumulator = FeatureStatisticsAccumulator(raw_harmonic, config.feature.rms_floor)
    valid_frames = 0
    for batch in loader:
        remaining = args.frames - valid_frames
        mask = batch["mask"]
        if int(mask.sum()) > remaining:
            flat_valid = mask.flatten().nonzero().flatten()[:remaining]
            limited = torch.zeros_like(mask.flatten())
            limited[flat_valid] = True
            mask = limited.view_as(mask)
        accumulator.update(
            batch["f0"].to(device, non_blocking=True),
            batch["rms"].to(device, non_blocking=True),
            mask.to(device, non_blocking=True),
        )
        valid_frames += int(mask.sum())
        if valid_frames >= args.frames:
            break
    manifest_hash = manifest_sha256(args.manifest)
    contract = accumulator.finalize(
        centers.cpu(),
        {
            "contract_version": 1,
            "mel_scale": config.feature.mel_scale,
            "mel_norm": config.feature.mel_norm,
            "hop_length": config.feature.hop_length,
            "n_fft": config.feature.n_fft,
            "win_length": config.feature.win_length,
            "power": config.feature.power,
            "center": config.feature.center,
            "pad_mode": config.feature.pad_mode,
            "log_base": config.feature.log_base,
            "log_clamp": config.feature.log_clamp,
            "mel_fmin": config.harmonic.fmin,
            "mel_fmax": config.harmonic.fmax,
            "sample_rate": config.harmonic.sample_rate,
            "f0_min": config.harmonic.f0_min,
            "f0_max": config.harmonic.f0_max,
            "nyquist_ratio": config.harmonic.nyquist_ratio,
            "narrow_bandwidth_semitones": (config.harmonic.narrow_bandwidth_semitones),
            "wide_bandwidth_semitones": config.harmonic.wide_bandwidth_semitones,
            "harmonic_weighting": "n^-0.5",
            "occupancy_normalization": "per_frame_max",
            "distance_definition": "signed_semitones_clipped_6_divided_6",
            "index_definition": "log1p_n_divided_log1p_global_nmax",
            "waveform_amplitude_convention": (
                config.feature.waveform_amplitude_convention
            ),
            "rms_definition": config.feature.rms_definition,
            "dataset_manifest_hash": manifest_hash,
            "exposure_semantics_hash": exposure_semantics_hash(config, manifest_hash),
            "stats_run_hash": stats_run_hash(
                seed=config.sampling.seed,
                frame_count=args.frames,
                stream="feature_fit",
            ),
        },
    )
    digest = contract.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                **contract.metadata,
                "harmonic_mean": contract.harmonic_mean.tolist(),
                "harmonic_std": contract.harmonic_std.tolist(),
                "rms_floor": contract.rms_floor,
                "rms_log_mean": contract.rms_log_mean,
                "rms_log_std": contract.rms_log_std,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
