from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .config import HARPConfig
from .manifest import ManifestEntry


@dataclass(frozen=True)
class SampleRequest:
    index: int
    frames: int
    seed: int


def bounded_normalize(
    weights: Sequence[float], lower: float, upper: float
) -> list[float]:
    values = [float(value) for value in weights]
    count = len(values)
    if not count or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("bounded normalization requires finite positive weights")
    if count * lower > 1 + 1e-12 or count * upper < 1 - 1e-12:
        raise ValueError("bounded normalization constraints are infeasible")
    result: list[float | None] = [None] * count
    free = set(range(count))
    remaining = 1.0
    while free:
        scale = remaining / sum(values[index] for index in free)
        low = [index for index in free if values[index] * scale < lower]
        high = [index for index in free if values[index] * scale > upper]
        if not low and not high:
            for index in free:
                result[index] = values[index] * scale
            break
        for index in low:
            result[index] = lower
            remaining -= lower
            free.remove(index)
        for index in high:
            result[index] = upper
            remaining -= upper
            free.remove(index)
    normalized = [float(value) for value in result]
    normalized[0] += 1.0 - sum(normalized)
    return normalized


class FeatureDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        mel_channels: int,
        content_dim: int,
        *,
        speaker_to_id: dict[str, int] | None = None,
        voiced_crop_probability: float = 0.7,
        mel_only: bool = False,
    ) -> None:
        self.entries = list(entries)
        self.mel_channels = mel_channels
        self.content_dim = content_dim
        self.voiced_crop_probability = voiced_crop_probability
        self.mel_only = mel_only
        speakers = sorted({entry.speaker_key for entry in entries})
        self.speaker_to_id = speaker_to_id or {
            speaker: index for index, speaker in enumerate(speakers)
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, request: int | SampleRequest) -> dict[str, Tensor]:
        if isinstance(request, int):
            request = SampleRequest(request, self.entries[request].frames, request)
        entry = self.entries[request.index]
        features = self._load(entry)
        available = min(value.shape[0] for value in features.values())
        wanted = min(request.frames, available)
        rng = random.Random(request.seed)
        starts = [rng.randrange(available - wanted + 1) for _ in range(8)]
        voiced_start = max(
            starts,
            key=lambda start: float(
                (features["f0"][start : start + wanted] > 0).float().mean()
            ),
        )
        start = (
            voiced_start
            if rng.random() < self.voiced_crop_probability
            else rng.choice(starts)
        )
        cropped = {
            name: value[start : start + wanted] for name, value in features.items()
        }
        return {
            **cropped,
            "speaker": torch.tensor(self.speaker_to_id[entry.speaker_key]),
            "length": torch.tensor(wanted),
            "requested_length": torch.tensor(request.frames),
        }

    def _load(self, entry: ManifestEntry) -> dict[str, Tensor]:
        prefix = Path(entry.feature_prefix)
        mel = _matrix(
            torch.load(f"{prefix}.mel.pt", map_location="cpu", weights_only=True),
            self.mel_channels,
        )
        f0 = _vector(
            torch.load(f"{prefix}.f0.pt", map_location="cpu", weights_only=True)
        )
        rms = _vector(
            torch.load(f"{prefix}.rms.pt", map_location="cpu", weights_only=True)
        )
        result = {"mel": mel.float(), "f0": f0.float(), "rms": rms.float()}
        if not self.mel_only:
            content_path = entry.content_feature_path or f"{prefix}.content.pt"
            content = _matrix(
                torch.load(content_path, map_location="cpu", weights_only=True),
                self.content_dim,
            )
            result["content"] = _resize(content.float(), mel.shape[0])
        return result


class HierarchicalBatchSampler(Sampler[list[SampleRequest]]):
    def __init__(self, entries: Sequence[ManifestEntry], config: HARPConfig) -> None:
        self.entries = entries
        sampling = config.sampling
        training = config.training
        self.batch_size = sampling.batch_size
        self.batch_frame_budget = sampling.batch_frame_budget
        self.steps_per_epoch = sampling.steps_per_epoch
        self.frame_buckets = training.frame_buckets
        self.bucket_probabilities = training.bucket_probabilities
        self.seed = sampling.seed
        self.epoch = 0
        hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, entry in enumerate(entries):
            if entry.split == "train" and entry.quality_status == "accepted":
                hierarchy[entry.dataset][entry.speaker][entry.song].append(index)
        self.hierarchy = hierarchy
        self.datasets = sorted(hierarchy)
        if set(self.datasets) != set(sampling.dataset_probabilities):
            raise ValueError("sampler datasets do not match configured probabilities")
        self.dataset_probabilities = [
            sampling.dataset_probabilities[name] for name in self.datasets
        ]
        self.speaker_probabilities = {}
        self.song_probabilities = {}
        for dataset, speakers in hierarchy.items():
            names = sorted(speakers)
            durations = [
                sum(
                    entries[index].frames
                    for song in speakers[name].values()
                    for index in song
                )
                for name in names
            ]
            probabilities = bounded_normalize(
                [value**sampling.speaker_duration_exponent for value in durations],
                sampling.speaker_probability_floor_ratio / len(names),
                sampling.speaker_probability_ceiling_ratio / len(names),
            )
            self.speaker_probabilities[dataset] = dict(
                zip(names, probabilities, strict=True)
            )
            self.song_probabilities[dataset] = {}
            for speaker, songs in speakers.items():
                song_names = sorted(songs)
                song_durations = [
                    sum(entries[index].frames for index in songs[name])
                    for name in song_names
                ]
                song_probabilities = bounded_normalize(
                    [
                        value**sampling.song_duration_exponent
                        for value in song_durations
                    ],
                    sampling.song_probability_floor_ratio / len(song_names),
                    sampling.song_probability_ceiling_ratio / len(song_names),
                )
                self.song_probabilities[dataset][speaker] = dict(
                    zip(song_names, song_probabilities, strict=True)
                )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[SampleRequest]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        for step in range(self.steps_per_epoch):
            frames = rng.choices(self.frame_buckets, self.bucket_probabilities, k=1)[0]
            batch_size = min(self.batch_size, self.batch_frame_budget // frames)
            batch = []
            for position in range(batch_size):
                dataset = rng.choices(self.datasets, self.dataset_probabilities, k=1)[0]
                speakers = sorted(self.hierarchy[dataset])
                speaker = rng.choices(
                    speakers,
                    [self.speaker_probabilities[dataset][name] for name in speakers],
                    k=1,
                )[0]
                songs = sorted(self.hierarchy[dataset][speaker])
                song = rng.choices(
                    songs,
                    [self.song_probabilities[dataset][speaker][name] for name in songs],
                    k=1,
                )[0]
                candidates = self.hierarchy[dataset][speaker][song]
                index = rng.choices(
                    candidates,
                    [self.entries[item].frames for item in candidates],
                    k=1,
                )[0]
                seed = (
                    self.seed + self.epoch * 10**9 + step * self.batch_size + position
                )
                batch.append(SampleRequest(index, frames, seed))
            yield batch


def collate_features(samples: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    maximum = max(int(sample["length"]) for sample in samples)
    result = {}
    for name in samples[0]:
        if name in {"speaker", "length", "requested_length"}:
            result[name] = torch.stack([sample[name] for sample in samples])
        else:
            result[name] = torch.stack(
                [
                    F.pad(sample[name], (0, 0, 0, maximum - sample[name].shape[0]))
                    for sample in samples
                ]
            )
    lengths = result["length"]
    result["mask"] = torch.arange(maximum)[None] < lengths[:, None]
    return result


def _matrix(value: Tensor, channels: int) -> Tensor:
    if value.ndim != 2:
        raise ValueError("feature matrix must be rank two")
    if value.shape[-1] == channels:
        return value
    if value.shape[0] == channels:
        return value.T
    raise ValueError(f"feature has no axis of size {channels}")


def _vector(value: Tensor) -> Tensor:
    if value.ndim == 1:
        return value[:, None]
    if value.ndim == 2 and 1 in value.shape:
        return value.reshape(-1, 1)
    raise ValueError("scalar feature must be a vector")


def _resize(value: Tensor, frames: int) -> Tensor:
    if value.shape[0] == frames:
        return value
    return F.interpolate(
        value.T[None], size=frames, mode="linear", align_corners=False
    )[0].T
