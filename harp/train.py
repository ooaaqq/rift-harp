from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import checkpoint_contract, validate_checkpoint_contract
from .config import HARPConfig
from .data import FeatureDataset, HierarchicalBatchSampler, collate_features
from .flow import HARPFlow
from .flow_transform import FlowTransform
from .manifest import load_manifest, manifest_sha256
from .model import HARPCore
from .optimizer import build_optimizer
from .performance import compile_model_in_place, configure_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RIFT-HARP foundation")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    config = HARPConfig.load(args.config)
    entries = load_manifest(args.manifest)
    train_entries = [
        entry
        for entry in entries
        if entry.split == "train" and entry.quality_status == "accepted"
    ]
    summary = {
        "train_recordings": len(train_entries),
        "speakers": len({entry.speaker_key for entry in train_entries}),
        "datasets": sorted({entry.dataset for entry in train_entries}),
        "config_sha256": config.digest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.execute_training:
        print("validation only; pass --execute-training to train")
        return
    train(config, train_entries, args)


def train(config: HARPConfig, entries: list, args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    runtime = configure_cuda(device)
    transform = FlowTransform.load(args.transform)
    transform_sha256 = _sha256(args.transform)
    if (
        transform.metadata.get("artifact_type") != "flow_transform_v1"
        or transform.metadata.get("contract_accepted") is not True
    ):
        raise ValueError("flow transform artifact has not passed the v1 contract")
    if transform.metadata.get("dataset_manifest_hash") != manifest_sha256(
        args.manifest
    ):
        raise ValueError("flow transform was fitted from a different manifest")
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        voiced_crop_probability=config.training.voiced_crop_probability,
    )
    if len(dataset.speaker_to_id) != config.num_speakers:
        raise ValueError("configured speaker count differs from training manifest")
    sampler = HierarchicalBatchSampler(entries, config)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_features,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = HARPCore(config.model, config.harmonic, config.num_speakers).to(device)
    system = HARPFlow(
        model,
        transform,
        speaker_drop_probability=config.training.speaker_drop_probability,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    )
    optimizer = build_optimizer(model, fused=device.type == "cuda")
    base_rates = [group["lr"] for group in optimizer.param_groups]
    contract = checkpoint_contract(config, model, transform_sha256)
    if not args.no_compile:
        compile_model_in_place(model, config.training.compile_mode)
        runtime["compiled"] = True
    else:
        runtime["compiled"] = False
    step, epoch, batch_offset = 0, 0, 0
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(payload["contract"], config)
        if payload["contract"]["flow_transform_sha256"] != transform_sha256:
            raise ValueError("resume flow transform differs")
        if payload["speaker_to_id"] != dataset.speaker_to_id:
            raise ValueError("resume speaker mapping differs")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        ema = {name: value.to(device) for name, value in payload["ema"].items()}
        step = int(payload["step"])
        epoch = int(payload["epoch"])
        batch_offset = int(payload["batch_offset"])
        torch.set_rng_state(payload["torch_rng_state"])
        if payload["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    target_steps = args.steps or config.training.max_steps
    if target_steps > config.training.max_steps or target_steps <= step:
        raise ValueError("invalid target step")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(
        json.dumps({"runtime": runtime, "contract": contract}, indent=2) + "\n"
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logged_frames = 0
    started = time.perf_counter()
    while step < target_steps:
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index < batch_offset:
                continue
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.items()
            }
            scale = min(1.0, (step + 1) / config.training.warmup_steps)
            for group, base_rate in zip(
                optimizer.param_groups, base_rates, strict=True
            ):
                group["lr"] = base_rate * scale
            loss = system(batch)
            _require_finite(loss.total, "loss", step)
            loss.total.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.grad_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _update_ema(ema, model, config.training.ema_decay)
            step += 1
            batch_offset = batch_index + 1
            logged_frames += int(batch["length"].sum())
            if step % config.training.log_every_steps == 0 or step == target_steps:
                elapsed = time.perf_counter() - started
                event = {
                    "step": step,
                    "loss": float(loss.total.detach()),
                    "grad_norm": float(grad_norm),
                    "frames_per_second": logged_frames / elapsed,
                    "lambda_floor_fraction": loss.lambda_floor_fraction,
                    "q_floor_fraction": loss.q_floor_fraction,
                    "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                    "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                }
                print(json.dumps(event), flush=True)
                started, logged_frames = time.perf_counter(), 0
            if (
                step % config.training.checkpoint_every_steps == 0
                or step == target_steps
            ):
                _save_checkpoint(
                    args.output / f"step-{step:07d}.pt",
                    contract,
                    model,
                    optimizer,
                    ema,
                    step,
                    epoch,
                    batch_offset,
                    dataset.speaker_to_id,
                )
            if step >= target_steps:
                return
        epoch += 1
        batch_offset = 0


@torch.no_grad()
def _update_ema(ema: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema[name].lerp_(value.detach(), 1 - decay)
        else:
            ema[name].copy_(value)


def _save_checkpoint(
    path: Path,
    contract: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, torch.Tensor],
    step: int,
    epoch: int,
    batch_offset: int,
    speaker_to_id: dict[str, int],
) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        torch.save(
            {
                "contract": contract,
                "step": step,
                "epoch": epoch,
                "batch_offset": batch_offset,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema,
                "speaker_to_id": speaker_to_id,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            temporary,
        )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _require_finite(value: torch.Tensor, name: str, step: int) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"non-finite {name} at step {step}")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
