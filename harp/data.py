import math
import random
import statistics
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
    start: int | None = None
    content_feature_path: str | None = None
    is_pseudo: bool = False


def bounded_normalize(
    weights: Sequence[float], lower: float, upper: float
) -> list[float]:
    values = [float(value) for value in weights]
    count = len(values)
    if not count or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("bounded normalization requires finite positive weights")
    if count * lower > 1 + 1e-12 or count * upper < 1 - 1e-12:
        raise ValueError("bounded normalization constraints are infeasible")

    # Solve sum(clip(c * w_i, lower, upper)) = 1.  The left hand side is
    # monotone in c, so bisection avoids assigning residual mass to an
    # arbitrary (sorted-first) item.
    def total(scale: float) -> float:
        return sum(min(upper, max(lower, scale * value)) for value in values)

    low_scale = 0.0
    high_scale = max(1.0, 1.0 / min(values))
    while total(high_scale) < 1.0:
        high_scale *= 2.0
    for _ in range(80):
        mid = (low_scale + high_scale) / 2.0
        if total(mid) < 1.0:
            low_scale = mid
        else:
            high_scale = mid
    normalized = [min(upper, max(lower, high_scale * value)) for value in values]
    # Only absorb sub-ulp error, and do so proportionally among non-boundary
    # entries rather than violating a configured bound.
    residual = 1.0 - sum(normalized)
    if abs(residual) > 1e-12:
        free = [
            index
            for index, value in enumerate(normalized)
            if lower + 1e-12 < value < upper - 1e-12
        ]
        if not free:
            raise RuntimeError("bounded normalization failed to resolve residual")
        share = residual / len(free)
        for index in free:
            normalized[index] += share
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
        features = self._load(entry, request.content_feature_path)
        available = features["mel"].shape[0]
        wanted = min(request.frames, available)
        rng = random.Random(request.seed)
        if request.start is None:
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
        else:
            start = request.start
            if not 0 <= start <= available - wanted:
                raise ValueError("fixed crop start is outside the recording")
        cropped = {
            name: value[start : start + wanted] for name, value in features.items()
        }
        return {
            **cropped,
            "speaker": torch.tensor(self.speaker_to_id[entry.speaker_key]),
            "length": torch.tensor(wanted),
            "requested_length": torch.tensor(request.frames),
            "entry_index": torch.tensor(request.index),
            "crop_start": torch.tensor(start),
            "is_pseudo": torch.tensor(request.is_pseudo),
        }

    def _load(
        self, entry: ManifestEntry, content_feature_path: str | None = None
    ) -> dict[str, Tensor]:
        prefix = Path(entry.feature_prefix)
        mel = _matrix(
            torch.load(f"{prefix}.mel.pt", map_location="cpu", weights_only=True),
            self.mel_channels,
        )
        if mel.shape[0] != entry.frames:
            raise ValueError(
                f"{entry.id}: manifest frames {entry.frames} differ from mel "
                f"frames {mel.shape[0]}"
            )
        f0 = _vector(
            torch.load(f"{prefix}.f0.pt", map_location="cpu", weights_only=True)
        )
        rms = _vector(
            torch.load(f"{prefix}.rms.pt", map_location="cpu", weights_only=True)
        )
        result = {"mel": mel.float(), "f0": f0.float(), "rms": rms.float()}
        if f0.shape[0] != mel.shape[0] or rms.shape[0] != mel.shape[0]:
            raise ValueError(
                f"{entry.id}: F0/RMS must match mel frames "
                f"({f0.shape[0]}/{rms.shape[0]} vs {mel.shape[0]})"
            )
        if not torch.isfinite(result["mel"]).all():
            raise ValueError(f"{entry.id}: mel contains non-finite values")
        if not torch.isfinite(result["rms"]).all() or bool((result["rms"] < 0).any()):
            raise ValueError(f"{entry.id}: RMS must be finite and nonnegative")
        if bool(torch.isinf(result["f0"]).any()):
            raise ValueError(f"{entry.id}: F0 contains infinite values")
        if not self.mel_only:
            content_path = (
                content_feature_path
                or entry.content_feature_path
                or f"{prefix}.content.pt"
            )
            content = _matrix(
                torch.load(content_path, map_location="cpu", weights_only=True),
                self.content_dim,
            )
            result["content"] = _resize(content.float(), mel.shape[0])
            if not torch.isfinite(result["content"]).all():
                raise ValueError(f"{entry.id}: content contains non-finite values")
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
        self.configured_dataset_families = sampling.dataset_families
        self.epoch = 0
        self.start_step = 0
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
        self._validate_exposure_constraints(config)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        if not 0 <= start_step <= self.steps_per_epoch:
            raise ValueError("sampler start step is outside the epoch")
        self.epoch = epoch
        self.start_step = start_step

    def __iter__(self) -> Iterator[list[SampleRequest]]:
        for step in range(self.start_step, self.steps_per_epoch):
            # A batch is a pure function of (seed, epoch, step). This preserves
            # the exact request stream when resuming without replaying prefetched
            # batches from the beginning of the epoch.
            rng = random.Random(self.seed + self.epoch * 1_000_003 + step * 97_003)
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

    def _validate_exposure_constraints(self, config: HARPConfig) -> None:
        sampling = config.sampling
        dataset_weights = dict(
            zip(self.datasets, self.dataset_probabilities, strict=True)
        )
        family_totals: dict[str, float] = defaultdict(float)
        for dataset, probability in dataset_weights.items():
            family = sampling.dataset_families.get(dataset, dataset)
            family_totals[family] += probability
        exceeded = {
            family: family_totals.get(family, 0.0)
            for family, cap in sampling.family_probability_caps.items()
            if family_totals.get(family, 0.0) > cap + 1e-12
        }
        if exceeded:
            raise ValueError(f"dataset family probability cap exceeded: {exceeded}")
        real_probabilities = [
            dataset_weights[dataset] * probability
            for dataset, speakers in self.speaker_probabilities.items()
            if dataset not in sampling.synthetic_datasets
            for probability in speakers.values()
        ]
        if real_probabilities:
            median = statistics.median(real_probabilities)
            singleton = [
                dataset_weights[dataset]
                for dataset, speakers in self.speaker_probabilities.items()
                if dataset not in sampling.synthetic_datasets and len(speakers) == 1
            ]
            maximum = max(singleton, default=0.0)
            limit = median * sampling.max_singleton_real_speaker_median_ratio
            if maximum > limit + 1e-12:
                raise ValueError(
                    "singleton real speaker probability exceeds median cap: "
                    f"{maximum:.8f} > {limit:.8f}"
                )

    def sampling_audit(self, max_steps: int) -> dict[str, object]:
        dataset_weights = dict(
            zip(self.datasets, self.dataset_probabilities, strict=True)
        )
        expected_crops_per_step = sum(
            probability * min(self.batch_size, self.batch_frame_budget // frames)
            for frames, probability in zip(
                self.frame_buckets, self.bucket_probabilities, strict=True
            )
        )
        speakers = []
        songs = []
        recordings = []
        for dataset in self.datasets:
            for speaker, speaker_probability in self.speaker_probabilities[
                dataset
            ].items():
                marginal = dataset_weights[dataset] * speaker_probability
                speakers.append(
                    {
                        "dataset": dataset,
                        "speaker": speaker,
                        "probability": marginal,
                        "expected_crops": marginal
                        * expected_crops_per_step
                        * max_steps,
                    }
                )
                for song, song_probability in self.song_probabilities[dataset][
                    speaker
                ].items():
                    song_marginal = marginal * song_probability
                    songs.append(
                        {
                            "dataset": dataset,
                            "speaker": speaker,
                            "song": song,
                            "probability": song_marginal,
                            "expected_crops": (
                                song_marginal * expected_crops_per_step * max_steps
                            ),
                        }
                    )
                    candidates = self.hierarchy[dataset][speaker][song]
                    total_frames = sum(
                        self.entries[index].frames for index in candidates
                    )
                    for index in candidates:
                        entry = self.entries[index]
                        recording_marginal = song_marginal * entry.frames / total_frames
                        recordings.append(
                            {
                                "dataset": dataset,
                                "speaker": speaker,
                                "song": song,
                                "recording_id": entry.id,
                                "source_frames": entry.frames,
                                "probability": recording_marginal,
                                "expected_crops": (
                                    recording_marginal
                                    * expected_crops_per_step
                                    * max_steps
                                ),
                            }
                        )
        expected_valid_frames_per_step = sum(
            recording["probability"]
            * sum(
                bucket_probability
                * min(self.batch_size, self.batch_frame_budget // bucket)
                * min(int(recording["source_frames"]), bucket)
                for bucket, bucket_probability in zip(
                    self.frame_buckets, self.bucket_probabilities, strict=True
                )
            )
            for recording in recordings
        )
        return {
            "artifact_type": "harp_sampling_audit_v1",
            "sampling_implementation": "hierarchical_sampler_step_keyed_v2",
            "dataset_probabilities": dataset_weights,
            "canonical_batches": {
                str(frames): min(self.batch_size, self.batch_frame_budget // frames)
                for frames in self.frame_buckets
            },
            "expected_crops_per_step": expected_crops_per_step,
            "expected_requested_frames_per_step": sum(
                probability
                * min(self.batch_size, self.batch_frame_budget // frames)
                * frames
                for frames, probability in zip(
                    self.frame_buckets, self.bucket_probabilities, strict=True
                )
            ),
            "expected_valid_frames_per_step": expected_valid_frames_per_step,
            "dataset_families": dict(self.configured_dataset_families),
            "speakers": speakers,
            "songs": songs,
            "recordings": recordings,
        }


def collate_features(samples: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    requested = {int(sample["requested_length"]) for sample in samples}
    if len(requested) != 1:
        raise ValueError("a canonical batch must contain exactly one requested bucket")
    maximum = requested.pop()
    result = {}
    for name in samples[0]:
        if name in {
            "speaker",
            "length",
            "requested_length",
            "entry_index",
            "crop_start",
            "noise_seed",
            "is_pseudo",
        }:
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
