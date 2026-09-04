import hashlib
import json
import subprocess

import pytest

from harp.config import HARPConfig
from harp.vocoder import PCNSFContract


def _config() -> HARPConfig:
    return HARPConfig(num_speakers=3)


def _lock(checkpoint_sha256: str) -> dict:
    config = _config()
    return {
        "schema_version": 1,
        "code": {"revision": "abc"},
        "checkpoint": {"extracted_sha256": checkpoint_sha256},
        "feature_contract": {
            "sample_rate": config.feature.sample_rate,
            "hop_length": config.feature.hop_length,
            "n_fft": config.feature.n_fft,
            "win_length": config.feature.win_length,
            "mel_channels": config.model.mel_channels,
            "fmin": config.harmonic.fmin,
            "fmax": config.harmonic.fmax,
            "log_base": "e",
            "mel_scale": config.feature.mel_scale,
            "mel_norm": config.feature.mel_norm,
            "power": config.feature.power,
            "center": config.feature.center,
            "pad_mode": config.feature.pad_mode,
            "log_clamp": config.feature.log_clamp,
        },
        "training_policy": {"mini_nsf": True, "pc_aug": True},
    }


def test_pc_nsf_contract_accepts_harp_feature_geometry(tmp_path) -> None:
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(_lock("0" * 64)))
    PCNSFContract.load(lock_path).validate(_config())


def test_pc_nsf_contract_rejects_feature_drift(tmp_path) -> None:
    payload = _lock("0" * 64)
    payload["feature_contract"]["fmax"] = 15_000
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fmax"):
        PCNSFContract.load(lock_path).validate(_config())


def test_pc_nsf_contract_verifies_checkout_and_checkpoint(tmp_path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=checkout, check=True)
    (checkout / "tracked").write_text("clean")
    subprocess.run(["git", "add", "tracked"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=checkout, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    payload = _lock(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    payload["code"]["revision"] = revision
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(payload))
    PCNSFContract.load(lock_path).verify_assets(checkout, checkpoint)
