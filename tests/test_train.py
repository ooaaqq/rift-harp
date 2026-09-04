import argparse
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from harp.config import (
    FeatureConfig,
    HarmonicConfig,
    HARPConfig,
    ModelConfig,
    SamplingConfig,
    TrainingConfig,
)
from harp.contracts import exposure_semantics_hash
from harp.feature_contract import neutral_feature_contract
from harp.flow_transform import FlowTransform
from harp.manifest import ManifestEntry
from harp.panel_cli import feature_sha256
from harp.train import train


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _setup(tmp_path: Path) -> tuple[HARPConfig, list[ManifestEntry], Path]:
    prefix = tmp_path / "features"
    frames = 3
    torch.save(torch.randn(frames, 4), f"{prefix}.mel.pt")
    torch.save(torch.randn(frames, 8), f"{prefix}.content.pt")
    torch.save(torch.tensor([110.0, 120.0, 0.0]), f"{prefix}.f0.pt")
    torch.save(torch.tensor([0.1, 0.2, 0.0]), f"{prefix}.rms.pt")
    entry = ManifestEntry(
        id="recording",
        dataset="test",
        speaker="voice",
        song="song",
        feature_prefix=str(prefix),
        frames=frames,
        split="train",
        quality_status="accepted",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": entry.id,
                "dataset": entry.dataset,
                "speaker": entry.speaker,
                "song": entry.song,
                "feature_prefix": entry.feature_prefix,
                "frames": entry.frames,
                "split": entry.split,
                "quality_status": entry.quality_status,
            }
        )
        + "\n"
    )
    transform_path = tmp_path / "flow.npz"
    feature_path = tmp_path / "feature.npz"
    audit_path = tmp_path / "flow.audit.json"
    config = HARPConfig(
        num_speakers=1,
        model=ModelConfig(
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
        ),
        harmonic=HarmonicConfig(sample_rate=16000, fmin=40, fmax=7600),
        feature=FeatureConfig(sample_rate=16000),
        training=TrainingConfig(
            warmup_valid_frames=12,
            max_steps=2,
            frame_buckets=(4,),
            bucket_probabilities=(1.0,),
            log_every_steps=1,
            telemetry_every_steps=1,
            audit_checkpoint_every_valid_frames=6,
            full_checkpoint_every_valid_frames=6,
            health_every_valid_frames=6,
            local_audit_every_valid_frames=6,
            endpoint_every_valid_frames=12,
            full_panel_every_valid_frames=12,
        ),
        sampling=SamplingConfig(
            dataset_probabilities={"test": 1.0},
            batch_size=2,
            batch_frame_budget=8,
            steps_per_epoch=2,
            num_workers=1,
            prefetch_factor=1,
        ),
    )
    config = dataclasses.replace(
        config,
        flow=dataclasses.replace(
            config.flow,
            transform_path=str(transform_path),
            audit_path=str(audit_path),
        ),
        feature=dataclasses.replace(config.feature, contract_path=str(feature_path)),
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    transform = FlowTransform(
        mean=torch.zeros(4),
        basis=torch.eye(4),
        gain=torch.ones(4),
        lambda_raw=torch.ones(4),
        lambda_effective=torch.ones(4),
        metadata={
            "artifact_type": "flow_transform_v1",
            "contract_accepted": True,
            "dataset_manifest_hash": manifest_hash,
        },
    )
    transform_hash = transform.save(transform_path)
    feature = neutral_feature_contract(4, 40, 7600)
    feature.metadata.update(
        {
            "artifact_type": "feature_contract_v1",
            "mel_scale": "slaney",
            "mel_norm": "slaney",
            "hop_length": 512,
            "n_fft": 2048,
            "win_length": 2048,
            "power": 1.0,
            "center": False,
            "pad_mode": "reflect",
            "log_base": "natural",
            "log_clamp": 1e-9,
            "mel_fmin": 40,
            "mel_fmax": 7600,
            "sample_rate": 16000,
            "f0_min": 40,
            "f0_max": 1600,
            "nyquist_ratio": 0.95,
            "narrow_bandwidth_semitones": 0.35,
            "wide_bandwidth_semitones": 1.0,
            "harmonic_weighting": "n^-0.5",
            "occupancy_normalization": "per_frame_max",
            "distance_definition": "signed_semitones_clipped_6_divided_6",
            "index_definition": "log1p_n_divided_log1p_global_nmax",
            "waveform_amplitude_convention": "float_-1_to_1",
            "rms_definition": "waveform_frame_rms",
            "dataset_manifest_hash": manifest_hash,
            "exposure_semantics_hash": exposure_semantics_hash(config, manifest_hash),
        }
    )
    feature.save(feature_path)
    audit_path.write_text(
        json.dumps(
            {
                "contract_accepted": True,
                "flow_transform_sha256": transform_hash,
                "exposure_semantics_hash": exposure_semantics_hash(
                    config, manifest_hash
                ),
            }
        )
    )
    return config, [entry], manifest


def _args(
    manifest: Path, output: Path, *, steps: int, resume: Path | None = None
) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=manifest,
        output=output,
        resume=resume,
        steps=steps,
        device="cpu",
        no_compile=True,
    )


def _latest_full(output: Path) -> Path:
    return sorted(output.glob("full-*.pt"))[-1]


def test_full_checkpoint_resume_matches_uninterrupted_training(
    tmp_path: Path, monkeypatch
) -> None:
    config, entries, manifest = _setup(tmp_path)
    monkeypatch.setattr(
        "harp.train._git_metadata",
        lambda: {"commit": "test", "dirty": False, "diff_sha256": "0" * 64},
    )

    uninterrupted = tmp_path / "uninterrupted"
    _seed_all(42)
    train(config, entries, _args(manifest, uninterrupted, steps=2))
    uninterrupted_payload = torch.load(
        _latest_full(uninterrupted), map_location="cpu", weights_only=False
    )

    resumed = tmp_path / "resumed"
    _seed_all(42)
    train(config, entries, _args(manifest, resumed, steps=1))
    first = _latest_full(resumed)
    request = json.loads((resumed / "audit_requests.jsonl").read_text().splitlines()[0])
    assert request["audits"] == ["local"]
    assert request["checkpoint"] == first.name
    _seed_all(999)
    train(config, entries, _args(manifest, resumed, steps=2, resume=first))
    resumed_payload = torch.load(
        _latest_full(resumed), map_location="cpu", weights_only=False
    )

    assert resumed_payload["progress"] == uninterrupted_payload["progress"]
    for name, value in uninterrupted_payload["model"].items():
        torch.testing.assert_close(
            resumed_payload["model"][name], value, rtol=0, atol=0
        )
    for name, value in uninterrupted_payload["ema"].items():
        torch.testing.assert_close(resumed_payload["ema"][name], value, rtol=0, atol=0)
    assert (
        resumed_payload["optimizer"]["param_groups"]
        == uninterrupted_payload["optimizer"]["param_groups"]
    )
    for index, state in uninterrupted_payload["optimizer"]["state"].items():
        for name, value in state.items():
            actual = resumed_payload["optimizer"]["state"][index][name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            else:
                assert actual == value


def test_local_and_endpoint_audits_run_on_frozen_synthetic_panel(
    tmp_path: Path, monkeypatch
) -> None:
    from harp.endpoint_audit_cli import main as endpoint_main
    from harp.local_audit_cli import main as local_main

    config, entries, manifest = _setup(tmp_path)
    monkeypatch.setattr(
        "harp.train._git_metadata",
        lambda: {"commit": "test", "dirty": False, "diff_sha256": "0" * 64},
    )
    output = tmp_path / "run"
    _seed_all(7)
    train(config, entries, _args(manifest, output, steps=1))
    checkpoint = _latest_full(output)
    config_path = tmp_path / "config.json"
    config.save(config_path)
    panels = tmp_path / "panels.json"
    panels.write_text(
        json.dumps(
            {
                "artifact_type": "harp_fixed_panels_v1",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "panels": {
                    "train": [
                        {
                            "entry_id": entries[0].id,
                            "requested_frames": 4,
                            "crop_start": 0,
                            "noise_seed": 123,
                            "feature_sha256": feature_sha256(entries[0]),
                        }
                    ]
                },
            }
        )
    )
    local_output = tmp_path / "local.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "harp-audit-local-field",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest),
            "--panels",
            str(panels),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(local_output),
            "--device",
            "cpu",
        ],
    )
    local_main()
    assert json.loads(local_output.read_text())["results"]

    endpoint_output = tmp_path / "endpoint.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "harp-audit-endpoint",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest),
            "--panels",
            str(panels),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(endpoint_output),
            "--steps",
            "1",
            "--device",
            "cpu",
        ],
    )
    endpoint_main()
    assert json.loads(endpoint_output.read_text())["results"]
