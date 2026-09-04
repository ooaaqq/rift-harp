from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .config import HARPConfig


@dataclass(frozen=True)
class PCNSFContract:
    revision: str
    checkpoint_sha256: str
    feature_contract: dict[str, object]
    training_policy: dict[str, object]

    @classmethod
    def load(cls, path: str | Path) -> PCNSFContract:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported PC-NSF lock schema")
        return cls(
            revision=str(payload["code"]["revision"]),
            checkpoint_sha256=str(payload["checkpoint"]["extracted_sha256"]),
            feature_contract=dict(payload["feature_contract"]),
            training_policy=dict(payload["training_policy"]),
        )

    def validate(self, config: HARPConfig) -> None:
        expected = {
            "sample_rate": config.feature.sample_rate,
            "hop_length": config.feature.hop_length,
            "n_fft": config.feature.n_fft,
            "win_length": config.feature.win_length,
            "mel_channels": config.model.mel_channels,
            "fmin": config.harmonic.fmin,
            "fmax": config.harmonic.fmax,
            "log_base": "e" if config.feature.log_base == "natural" else None,
            "mel_scale": config.feature.mel_scale,
            "mel_norm": config.feature.mel_norm,
            "power": config.feature.power,
            "center": config.feature.center,
            "pad_mode": config.feature.pad_mode,
            "log_clamp": config.feature.log_clamp,
        }
        mismatched = [
            name
            for name, value in expected.items()
            if self.feature_contract.get(name) != value
        ]
        if mismatched:
            raise ValueError(f"PC-NSF feature contract differs in {mismatched}")
        required_policy = {"mini_nsf": True, "pc_aug": True}
        mismatched_policy = [
            name
            for name, value in required_policy.items()
            if self.training_policy.get(name) != value
        ]
        if mismatched_policy:
            raise ValueError(f"PC-NSF training policy differs in {mismatched_policy}")

    def verify_assets(self, checkout: Path, checkpoint: Path) -> None:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != self.revision:
            raise ValueError(f"PC-NSF checkout is {head}, expected {self.revision}")
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty:
            raise ValueError("PC-NSF checkout has uncommitted modifications")
        if _sha256(checkpoint) != self.checkpoint_sha256:
            raise ValueError("installed PC-NSF checkpoint SHA256 mismatch")


def load_pc_nsf(
    checkout: Path,
    checkpoint_path: Path,
    lock_path: Path,
    config: HARPConfig,
    device: torch.device,
):
    contract = PCNSFContract.load(lock_path)
    contract.validate(config)
    contract.verify_assets(checkout, checkpoint_path)
    module_path = checkout / "models/nsf_HiFigan/models.py"
    specification = importlib.util.spec_from_file_location("harp_pc_nsf", module_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load official PC-NSF module: {module_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    generator_config = {
        "mini_nsf": True,
        "noise_sigma": 0.0,
        "upsample_rates": [8, 8, 2, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4, 4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "resblock": "1",
        "sampling_rate": config.feature.sample_rate,
        "num_mels": config.model.mel_channels,
        "hop_size": config.feature.hop_length,
        "n_fft": config.feature.n_fft,
        "win_size": config.feature.win_length,
        "fmin": config.harmonic.fmin,
        "fmax": config.harmonic.fmax,
        "pc_aug": True,
    }
    generator = module.Generator(module.AttrDict(generator_config)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise ValueError("official PC-NSF checkpoint must contain state_dict")
    state = {
        name.removeprefix("generator."): value
        for name, value in checkpoint["state_dict"].items()
        if name.startswith("generator.")
    }
    if not state:
        raise ValueError("official PC-NSF checkpoint contains no generator weights")
    generator.load_state_dict(state, strict=True)
    generator.eval()
    generator.remove_weight_norm()
    return generator, contract


@torch.inference_mode()
def synthesize_pc_nsf(
    generator, mel: Tensor, f0: Tensor, device: torch.device
) -> Tensor:
    mel = torch.as_tensor(mel).float()
    f0 = torch.as_tensor(f0).float().flatten()
    if mel.ndim != 2:
        raise ValueError("mel must be rank two")
    if mel.shape[0] != f0.numel() and mel.shape[1] == f0.numel():
        mel = mel.T
    if mel.shape[0] != f0.numel():
        raise ValueError("mel and F0 frame counts differ")
    if not torch.isfinite(mel).all() or not torch.isfinite(f0).all():
        raise ValueError("mel or F0 contains non-finite values")
    if bool((f0 < 0).any()) or bool((f0 > 2000).any()):
        raise ValueError("F0 is outside [0, 2000] Hz")
    waveform = generator(mel.T[None].to(device), f0[None].to(device))
    waveform = waveform[0, 0].float().cpu()
    if not torch.isfinite(waveform).all():
        raise ValueError("PC-NSF produced a non-finite waveform")
    return waveform.clamp(-1, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
