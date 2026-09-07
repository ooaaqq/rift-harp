import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .config import HARPConfig
from .data import FeatureDataset, SampleRequest
from .manifest import ManifestEntry, load_manifest, manifest_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze HARP train/shadow panels")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-length", type=int, default=24)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    config = HARPConfig.load(args.config)
    entries = load_manifest(args.manifest)
    train_songs = {
        (entry.dataset, entry.song)
        for entry in entries
        if entry.split == "train" and entry.quality_status == "accepted"
    }
    shadow_songs = {
        (entry.dataset, entry.song)
        for entry in entries
        if entry.split in {"validation", "test"} and entry.quality_status == "accepted"
    }
    overlap = train_songs & shadow_songs
    if overlap:
        raise ValueError(f"manifest is not song-disjoint: {sorted(overlap)[:5]}")
    train_speaker_keys = sorted(
        {
            entry.speaker_key
            for entry in entries
            if entry.split == "train" and entry.quality_status == "accepted"
        }
    )
    speaker_to_id = {speaker: index for index, speaker in enumerate(train_speaker_keys)}
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        speaker_to_id=speaker_to_id,
        voiced_crop_probability=config.training.voiced_crop_probability,
    )
    panels = {
        "train": _build_panel(
            entries,
            dataset,
            {"train"},
            (256, 512),
            args.samples_per_length,
            args.seed,
            set(train_speaker_keys),
        ),
        "song_disjoint_shadow": _build_panel(
            entries,
            dataset,
            {"validation", "test"},
            (256, 512, 768),
            args.samples_per_length,
            args.seed + 1,
            set(train_speaker_keys),
        ),
    }
    payload = {
        "artifact_type": "harp_fixed_panels_v1",
        "manifest_sha256": manifest_sha256(args.manifest),
        "seed": args.seed,
        "samples_per_length": args.samples_per_length,
        "panels": panels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_panel(
    entries: list[ManifestEntry],
    dataset: FeatureDataset,
    splits: set[str],
    lengths: tuple[int, ...],
    samples_per_length: int,
    seed: int,
    allowed_speakers: set[str],
) -> list[dict[str, object]]:
    candidates = [
        index
        for index, entry in enumerate(entries)
        if entry.split in splits
        and entry.quality_status == "accepted"
        and entry.speaker_key in allowed_speakers
    ]
    if not candidates:
        raise ValueError(f"panel has no accepted entries for splits {sorted(splits)}")
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in candidates:
        entry = entries[index]
        grouped[(entry.dataset, entry.speaker)].append(index)
    rng = random.Random(seed)
    # Shuffle the stratification units once so coverage is independent of
    # dictionary/name ordering. Reuse the assignment stream across lengths.
    speaker_groups = sorted(grouped)
    rng.shuffle(speaker_groups)
    assignments = [
        speaker_groups[position % len(speaker_groups)]
        for position in range(samples_per_length * len(lengths))
    ]
    result = []
    assignment_index = 0
    for length in lengths:
        for position in range(samples_per_length):
            group = assignments[assignment_index]
            assignment_index += 1
            index = rng.choice(grouped[group])
            request_seed = seed + length * 100_003 + position
            sample = dataset[SampleRequest(index, length, request_seed)]
            entry = entries[index]
            result.append(
                {
                    "entry_id": entry.id,
                    "dataset": entry.dataset,
                    "speaker": entry.speaker,
                    "song": entry.song,
                    "requested_frames": length,
                    "actual_frames": int(sample["length"]),
                    "crop_start": int(sample["crop_start"]),
                    "noise_seed": request_seed + 9_000_001,
                    "feature_sha256": feature_sha256(entry),
                }
            )
    return result


def feature_sha256(entry: ManifestEntry) -> dict[str, str]:
    paths = {
        "mel": Path(f"{entry.feature_prefix}.mel.pt"),
        "f0": Path(f"{entry.feature_prefix}.f0.pt"),
        "rms": Path(f"{entry.feature_prefix}.rms.pt"),
        "content": Path(
            entry.content_feature_path or f"{entry.feature_prefix}.content.pt"
        ),
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def validate_panel_features(entry: ManifestEntry, item: dict[str, object]) -> None:
    if item.get("feature_sha256") != feature_sha256(entry):
        raise ValueError(f"fixed panel features changed for {entry.id}")


if __name__ == "__main__":
    main()
