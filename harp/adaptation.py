"""Target-specific HARP finetuning with pseudo-speaker content inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Sampler

from .checkpoint import validate_checkpoint_contract
from .config import HARPConfig
from .data import FeatureDataset, SampleRequest, collate_features
from .feature_contract import FeatureContract, validate_feature_contract
from .flow import HARPFlow
from .flow_transform import FlowTransform
from .manifest import ManifestEntry, load_manifest, manifest_sha256
from .model import HARPCore
from .performance import compile_model_in_place, configure_cuda
from .precision import configure_heavy_linears


@dataclass(frozen=True)
class PseudoVariant:
    carrier_speaker_key: str
    content_feature_path: str
    content_sha256: str
    quality_status: str


def load_pseudo_bank(
    path: str | Path,
) -> tuple[dict[str, list[PseudoVariant]], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "rift_harp_pseudo_content_bank_v1":
        raise ValueError("pseudo bank has an unsupported artifact type")
    grouped: dict[str, list[PseudoVariant]] = defaultdict(list)
    for item in payload.get("variants", []):
        variant = PseudoVariant(
            carrier_speaker_key=str(item["carrier_speaker_key"]),
            content_feature_path=str(item["content_feature_path"]),
            content_sha256=str(item["content_sha256"]),
            quality_status=str(item["quality_status"]),
        )
        if variant.quality_status == "accepted":
            grouped[str(item["origin_target_recording_id"])].append(variant)
    return dict(grouped), payload


def validate_pseudo_features(
    entries: list[ManifestEntry],
    variants: dict[str, list[PseudoVariant]],
    content_dim: int,
) -> dict[str, int]:
    entry_by_id = {entry.id: entry for entry in entries}
    accepted = 0
    recordings = 0
    for recording_id, choices in variants.items():
        if recording_id not in entry_by_id:
            raise ValueError(f"pseudo bank references unknown target {recording_id}")
        recordings += 1
        for variant in choices:
            path = Path(variant.content_feature_path)
            if _sha256(path) != variant.content_sha256:
                raise ValueError(f"pseudo content hash differs: {path}")
            value = torch.load(path, map_location="cpu", weights_only=True)
            expected = (entry_by_id[recording_id].frames, content_dim)
            if value.shape != expected or not bool(torch.isfinite(value).all()):
                raise ValueError(f"pseudo content is invalid: {path}")
            accepted += 1
    if not accepted:
        raise ValueError("pseudo bank contains no accepted content variants")
    return {"accepted_variants": accepted, "covered_recordings": recordings}


class PseudoPairedBatchSampler(Sampler[list[SampleRequest]]):
    """Sample target song and recording before choosing a content variant."""

    def __init__(
        self,
        entries: list[ManifestEntry],
        variants: dict[str, list[PseudoVariant]],
        config: HARPConfig,
        *,
        seed: int,
        pseudo_probability: float = 0.7,
        batch_frame_budget: int = 12_288,
    ) -> None:
        self.entries = entries
        self.variants = variants
        self.buckets = config.training.frame_buckets
        self.bucket_probabilities = config.training.bucket_probabilities
        self.steps_per_epoch = config.sampling.steps_per_epoch
        self.seed = seed
        self.pseudo_probability = pseudo_probability
        self.batch_frame_budget = batch_frame_budget
        self.epoch = 0
        self.start_step = 0
        songs: dict[str, list[int]] = defaultdict(list)
        for index, entry in enumerate(entries):
            if entry.split == "train" and entry.quality_status == "accepted":
                songs[entry.song].append(index)
        if not songs:
            raise ValueError("target manifest has no accepted train recordings")
        self.songs = sorted(songs)
        self.recordings = dict(songs)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        if not 0 <= start_step <= self.steps_per_epoch:
            raise ValueError("sampler start step is outside the epoch")
        self.epoch, self.start_step = epoch, start_step

    def __iter__(self) -> Iterator[list[SampleRequest]]:
        for step in range(self.start_step, self.steps_per_epoch):
            rng = random.Random(self.seed + self.epoch * 1_000_003 + step * 97_003)
            frames = rng.choices(self.buckets, self.bucket_probabilities, k=1)[0]
            batch_size = self.batch_frame_budget // frames
            batch = []
            for _ in range(batch_size):
                song = rng.choice(self.songs)
                candidates = self.recordings[song]
                index = rng.choices(
                    candidates,
                    [self.entries[item].frames for item in candidates],
                    k=1,
                )[0]
                available = self.variants.get(self.entries[index].id, [])
                use_pseudo = bool(available) and rng.random() < self.pseudo_probability
                variant = rng.choice(available) if use_pseudo else None
                batch.append(
                    SampleRequest(
                        index=index,
                        frames=frames,
                        seed=rng.randrange(2**63),
                        content_feature_path=(
                            variant.content_feature_path if variant else None
                        ),
                        is_pseudo=use_pseudo,
                    )
                )
            yield batch


def configure_finetune_parameters(model: HARPCore) -> tuple[list[str], list[str]]:
    trainable, frozen = [], []
    fixed = {"content_mix", "pitch_mix", "harmonic_mix", "energy_mix"}
    for name, parameter in model.named_parameters():
        should_train = not (name.startswith("speaker.") or name in fixed)
        parameter.requires_grad_(should_train)
        (trainable if should_train else frozen).append(name)
    return trainable, frozen


def build_finetune_optimizer(
    model: HARPCore,
    target_code: nn.Parameter,
    *,
    model_learning_rate: float,
    code_learning_rate: float,
    fused: bool,
) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        destination = (
            no_decay if parameter.ndim < 2 or name.endswith(".bias") else decay
        )
        destination.append(parameter)
    groups = [
        {
            "params": decay,
            "lr": model_learning_rate,
            "weight_decay": 0.01,
            "role": "model_decay",
        },
        {
            "params": no_decay,
            "lr": model_learning_rate,
            "weight_decay": 0.0,
            "role": "model_no_decay",
        },
        {
            "params": [target_code],
            "lr": code_learning_rate,
            "weight_decay": 0.0,
            "role": "target_code",
        },
    ]
    return torch.optim.AdamW(
        groups, betas=(0.9, 0.95), eps=1e-8, fused=fused, foreach=False
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parent(
    config: HARPConfig,
    checkpoint_path: Path,
    feature: FeatureContract,
    device: torch.device,
) -> tuple[HARPCore, dict]:
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if payload.get("checkpoint_type") not in {"full", "audit"}:
        raise ValueError("parent must be a HARP full or audit checkpoint")
    validate_checkpoint_contract(payload["contract"], config)
    model = HARPCore(config.model, config.harmonic, feature, config.num_speakers)
    configure_heavy_linears(model, config.model.heavy_linear_precision)
    model.load_state_dict(payload["ema"], strict=True)
    return model.to(device), payload


@torch.no_grad()
def _update_ema(
    model_ema: dict[str, Tensor],
    code_ema: Tensor,
    model: HARPCore,
    code: Tensor,
    decay: float,
) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            model_ema[name].lerp_(value.detach(), 1 - decay)
        else:
            model_ema[name].copy_(value)
    code_ema.lerp_(code.detach(), 1 - decay)


def _atomic_save(payload: dict, path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _save_checkpoint(
    path: Path,
    kind: str,
    run: dict,
    model: HARPCore,
    target_code: Tensor,
    model_ema: dict[str, Tensor],
    code_ema: Tensor,
    optimizer: torch.optim.Optimizer,
    progress: dict,
    epoch: int,
    batch_offset: int,
) -> None:
    payload = {
        "checkpoint_type": f"singer_finetune_{kind}_v1",
        "run": run,
        "model": model.state_dict(),
        "ema": model_ema,
        "target_code": target_code.detach(),
        "target_code_ema": code_ema,
        "progress": progress,
    }
    if kind == "full":
        payload.update(
            {
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "batch_offset": batch_offset,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
            }
        )
    _atomic_save(payload, path)


def finetune(config: HARPConfig, args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    configure_cuda(
        device,
        sdpa_backend=config.training.sdpa_backend,
        allow_tf32=config.training.allow_tf32,
    )
    entries = load_manifest(args.manifest)
    variants, bank = load_pseudo_bank(args.pseudo_bank)
    bank_summary = validate_pseudo_features(entries, variants, config.model.content_dim)
    manifest_hash = manifest_sha256(args.manifest)
    parent_path = Path(args.parent)
    if bank.get("target_manifest_sha256") != manifest_hash:
        raise ValueError("pseudo bank was built from a different target manifest")
    if bank.get("teacher_checkpoint_sha256") != _sha256(parent_path):
        raise ValueError("pseudo bank teacher differs from finetune parent")
    transform = FlowTransform.load(config.flow.transform_path).to(device)
    feature = FeatureContract.load(config.feature.contract_path)
    validate_feature_contract(feature, config.model, config.harmonic, config.feature)
    model, parent = _load_parent(config, parent_path, feature, device)
    trainable_names, frozen_names = configure_finetune_parameters(model)
    target_code = nn.Parameter(model.speaker.weight[:-1].detach().mean(dim=0).clone())
    optimizer = build_finetune_optimizer(
        model,
        target_code,
        model_learning_rate=args.model_learning_rate,
        code_learning_rate=args.code_learning_rate,
        fused=device.type == "cuda",
    )
    base_rates = [group["lr"] for group in optimizer.param_groups]
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        voiced_crop_probability=config.training.voiced_crop_probability,
    )
    sampler = PseudoPairedBatchSampler(
        entries,
        variants,
        config,
        seed=args.seed,
        pseudo_probability=args.pseudo_probability,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_features,
        num_workers=config.sampling.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=config.sampling.prefetch_factor,
        persistent_workers=config.sampling.persistent_workers,
    )
    system = HARPFlow(
        model,
        transform,
        speaker_drop_probability=0.0,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    )
    if not args.no_compile:
        compile_model_in_place(
            model,
            config.training.compile_mode,
            epilogue_fusion=config.training.inductor_epilogue_fusion,
            shape_padding=config.training.inductor_shape_padding,
        )
    output = Path(args.output)
    run = {
        "artifact_type": "rift_harp_singer_acoustic_finetune_v1",
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": _sha256(parent_path),
        "parent_state": "ema",
        "parent_progress": parent.get("progress"),
        "target_manifest": str(args.manifest),
        "target_manifest_sha256": manifest_hash,
        "pseudo_bank": str(args.pseudo_bank),
        "pseudo_bank_sha256": _sha256(args.pseudo_bank),
        "config_sha256": config.digest(),
        "pseudo_probability": args.pseudo_probability,
        "model_learning_rate": args.model_learning_rate,
        "code_learning_rate": args.code_learning_rate,
        "warmup_valid_frames": args.warmup_frames,
        "ema_half_life_valid_frames": args.ema_half_life_frames,
        "trainable_parameters": trainable_names + ["target_code"],
        "frozen_parameters": frozen_names,
        "created_at_unix": time.time(),
        **bank_summary,
    }
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError("resume output directory does not exist")
        recorded_run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        comparable = {
            key: value for key, value in run.items() if key != "created_at_unix"
        }
        recorded_comparable = {
            key: value
            for key, value in recorded_run.items()
            if key != "created_at_unix"
        }
        if comparable != recorded_comparable:
            raise ValueError("resume run contract differs")
        run = recorded_run
    else:
        if output.exists():
            raise FileExistsError("finetune output directory already exists")
        output.mkdir(parents=True)
        (output / "run.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    model_ema = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    code_ema = target_code.detach().clone()
    progress = {
        "global_step": 0,
        "seen_target_valid_frames": 0,
        "seen_original_valid_frames": 0,
        "seen_pseudo_valid_frames": 0,
        "seen_requested_frames": 0,
    }
    milestones = sorted(
        set(list(args.audit_frames) + list(args.full_frames) + [args.valid_frames])
    )
    next_milestone = 0
    epoch = 0
    batch_offset = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_type") != "singer_finetune_full_v1":
            raise ValueError("finetune can resume only from a full checkpoint")
        if payload.get("run") != run:
            raise ValueError("resume checkpoint belongs to another finetune run")
        model.load_state_dict(payload["model"], strict=True)
        target_code.data.copy_(payload["target_code"].to(device))
        optimizer.load_state_dict(payload["optimizer"])
        model_ema = {name: value.to(device) for name, value in payload["ema"].items()}
        code_ema = payload["target_code_ema"].to(device)
        progress = dict(payload["progress"])
        epoch = int(payload["epoch"])
        batch_offset = int(payload["batch_offset"])
        torch.set_rng_state(payload["torch_rng_state"])
        if payload["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        while (
            next_milestone < len(milestones)
            and milestones[next_milestone] <= progress["seen_target_valid_frames"]
        ):
            next_milestone += 1
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable.append(target_code)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started, interval_frames = time.perf_counter(), 0
    while progress["seen_target_valid_frames"] < args.valid_frames:
        sampler.set_epoch(epoch, batch_offset)
        for batch_index, batch in enumerate(loader, start=batch_offset):
            valid_by_sample = batch["length"].long()
            pseudo_by_sample = batch["is_pseudo"].bool()
            valid_frames = int(valid_by_sample.sum())
            pseudo_frames = int(valid_by_sample[pseudo_by_sample].sum())
            original_frames = valid_frames - pseudo_frames
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.items()
            }
            scale = min(
                1.0,
                (progress["seen_target_valid_frames"] + valid_frames)
                / args.warmup_frames,
            )
            for group, base_rate in zip(
                optimizer.param_groups, base_rates, strict=True
            ):
                group["lr"] = base_rate * scale
            code = target_code.unsqueeze(0).expand(batch["mel"].shape[0], -1)
            loss = system(batch, speaker_code_override=code)
            if not bool(torch.isfinite(loss.total.detach())):
                raise FloatingPointError("non-finite finetune loss")
            loss.total.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                trainable, 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            decay = 2.0 ** (-valid_frames / args.ema_half_life_frames)
            _update_ema(model_ema, code_ema, model, target_code, decay)
            progress["global_step"] += 1
            progress["seen_target_valid_frames"] += valid_frames
            progress["seen_original_valid_frames"] += original_frames
            progress["seen_pseudo_valid_frames"] += pseudo_frames
            progress["seen_requested_frames"] += int(batch["requested_length"].sum())
            interval_frames += valid_frames
            if progress["global_step"] % args.log_every_steps == 0:
                pseudo_mask = batch["is_pseudo"].bool()
                original_mask = ~pseudo_mask
                event = {
                    **progress,
                    "loss": float(loss.total.detach()),
                    "original_loss": (
                        float(loss.flow_by_sample[original_mask].mean().detach())
                        if bool(original_mask.any())
                        else None
                    ),
                    "pseudo_loss": (
                        float(loss.flow_by_sample[pseudo_mask].mean().detach())
                        if bool(pseudo_mask.any())
                        else None
                    ),
                    "actual_pseudo_frame_fraction": progress["seen_pseudo_valid_frames"]
                    / progress["seen_target_valid_frames"],
                    "warmup_scale": scale,
                    "ema_step_decay": decay,
                    "grad_norm": float(grad_norm),
                    "frames_per_second": interval_frames
                    / (time.perf_counter() - started),
                }
                print(json.dumps(event), flush=True)
                started, interval_frames = time.perf_counter(), 0
            while (
                next_milestone < len(milestones)
                and progress["seen_target_valid_frames"] >= milestones[next_milestone]
            ):
                threshold = milestones[next_milestone]
                kind = (
                    "full"
                    if threshold in args.full_frames or threshold == args.valid_frames
                    else "audit"
                )
                path = output / (
                    f"{kind}-step-{progress['global_step']:07d}"
                    f"-frames-{progress['seen_target_valid_frames']:012d}.pt"
                )
                _save_checkpoint(
                    path,
                    kind,
                    run,
                    model,
                    target_code,
                    model_ema,
                    code_ema,
                    optimizer,
                    progress,
                    epoch,
                    batch_index + 1,
                )
                print(json.dumps({"checkpoint": str(path), "kind": kind}), flush=True)
                next_milestone += 1
            if progress["seen_target_valid_frames"] >= args.valid_frames:
                return
        epoch += 1
        batch_offset = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pseudo-bank", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid-frames", type=int, default=100_000_000)
    parser.add_argument("--warmup-frames", type=int, default=2_000_000)
    parser.add_argument("--ema-half-life-frames", type=int, default=2_000_000)
    parser.add_argument("--model-learning-rate", type=float, default=2e-5)
    parser.add_argument("--code-learning-rate", type=float, default=1e-4)
    parser.add_argument("--pseudo-probability", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument(
        "--audit-frames", type=int, nargs="+", default=(5_000_000, 10_000_000)
    )
    parser.add_argument(
        "--full-frames",
        type=int,
        nargs="+",
        default=(20_000_000, 50_000_000, 100_000_000, 150_000_000),
    )
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.pseudo_probability <= 1:
        parser.error("--pseudo-probability must be in [0, 1]")
    if min(args.valid_frames, args.warmup_frames, args.ema_half_life_frames) <= 0:
        parser.error("frame counts must be positive")
    finetune(HARPConfig.load(args.config), args)


if __name__ == "__main__":
    main()
