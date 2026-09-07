import hashlib
import json
from typing import Any

from .config import HARPConfig


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def exposure_semantics(config: HARPConfig, manifest_sha256: str) -> dict[str, Any]:
    return {
        "version": 1,
        "manifest_sha256": manifest_sha256,
        "dataset_probabilities": config.sampling.dataset_probabilities,
        "dataset_families": config.sampling.dataset_families,
        "family_probability_caps": config.sampling.family_probability_caps,
        "synthetic_datasets": config.sampling.synthetic_datasets,
        "max_singleton_real_speaker_median_ratio": (
            config.sampling.max_singleton_real_speaker_median_ratio
        ),
        "speaker_duration_exponent": config.sampling.speaker_duration_exponent,
        "speaker_probability_floor_ratio": (
            config.sampling.speaker_probability_floor_ratio
        ),
        "speaker_probability_ceiling_ratio": (
            config.sampling.speaker_probability_ceiling_ratio
        ),
        "song_duration_exponent": config.sampling.song_duration_exponent,
        "song_probability_floor_ratio": config.sampling.song_probability_floor_ratio,
        "song_probability_ceiling_ratio": (
            config.sampling.song_probability_ceiling_ratio
        ),
        "frame_buckets": config.training.frame_buckets,
        "bucket_probabilities": config.training.bucket_probabilities,
        "crop_candidate_count": 8,
        "voiced_crop_probability": config.training.voiced_crop_probability,
        "short_recording_behavior": "use_all_available_then_pad_and_mask",
    }


def exposure_semantics_hash(config: HARPConfig, manifest_sha256: str) -> str:
    return json_sha256(exposure_semantics(config, manifest_sha256))


def batch_runtime(config: HARPConfig) -> dict[str, Any]:
    return {
        "version": 1,
        "batch_size": config.sampling.batch_size,
        "batch_frame_budget": config.sampling.batch_frame_budget,
        "num_workers": config.sampling.num_workers,
        "prefetch_factor": config.sampling.prefetch_factor,
        "persistent_workers": config.sampling.persistent_workers,
        "canonical_bucket_padding": config.sampling.canonical_bucket_padding,
        "canonical_shapes": {
            str(frames): [
                min(
                    config.sampling.batch_size,
                    config.sampling.batch_frame_budget // frames,
                ),
                frames,
            ]
            for frames in config.training.frame_buckets
        },
    }


def batch_runtime_hash(config: HARPConfig) -> str:
    return json_sha256(batch_runtime(config))


def stats_run_hash(*, seed: int, frame_count: int, stream: str) -> str:
    return json_sha256(
        {
            "version": 1,
            "seed": seed,
            "frame_count": frame_count,
            "stream": stream,
            "sampling_implementation": "hierarchical_sampler_step_keyed_v2",
        }
    )
