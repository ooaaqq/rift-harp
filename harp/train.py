from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import (
    checkpoint_contract,
    validate_checkpoint_contract,
    validate_contract_identity,
)
from .config import HARPConfig
from .contracts import (
    batch_runtime,
    batch_runtime_hash,
    exposure_semantics,
    exposure_semantics_hash,
    json_sha256,
)
from .data import FeatureDataset, HierarchicalBatchSampler, collate_features
from .feature_contract import FeatureContract, validate_feature_contract
from .flow import HARPFlow
from .flow_transform import FlowTransform
from .manifest import load_manifest, manifest_sha256
from .model import HARPCore
from .optimizer import build_optimizer
from .performance import compile_model_in_place, configure_cuda
from .precision import configure_heavy_linears
from .telemetry import model_activation_telemetry, optimizer_role_telemetry
from .training_state import (
    TrainingProgress,
    cadence_crossed,
    ema_decay_for_batch,
    ema_half_life_valid_frames,
    warmup_scale,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RIFT-HARP foundation")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda")
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
    runtime = configure_cuda(
        device,
        sdpa_backend=config.training.sdpa_backend,
        allow_tf32=config.training.allow_tf32,
    )
    runtime.update(_runtime_metadata())
    source_control = _git_metadata()
    if source_control["dirty"]:
        raise ValueError("foundation training requires a clean Git worktree")
    manifest_hash = manifest_sha256(args.manifest)
    exposure_hash = exposure_semantics_hash(config, manifest_hash)
    transform_path = Path(config.flow.transform_path)
    transform_audit_path = Path(config.flow.audit_path)
    feature_path = Path(config.feature.contract_path)
    transform = FlowTransform.load(transform_path).to(device)
    transform_sha256 = _sha256(transform_path)
    _load_transform_audit(transform_audit_path, transform_sha256, exposure_hash)
    transform_audit_sha256 = _sha256(transform_audit_path)
    feature_contract = FeatureContract.load(feature_path)
    validate_feature_contract(
        feature_contract, config.model, config.harmonic, config.feature
    )
    feature_contract_sha256 = _sha256(feature_path)
    if (
        transform.metadata.get("artifact_type") != "flow_transform_v1"
        or transform.metadata.get("contract_accepted") is not True
    ):
        raise ValueError("flow transform artifact has not passed the v1 contract")
    if transform.metadata.get("dataset_manifest_hash") != manifest_sha256(
        args.manifest
    ):
        raise ValueError("flow transform was fitted from a different manifest")
    if feature_contract.metadata.get("artifact_type") != "feature_contract_v1":
        raise ValueError("feature artifact is not feature_contract_v1")
    if feature_contract.metadata.get("dataset_manifest_hash") != manifest_hash:
        raise ValueError("feature contract was fitted from a different manifest")
    if feature_contract.metadata.get("exposure_semantics_hash") != exposure_hash:
        raise ValueError("feature contract exposure semantics differ from this run")
    if feature_contract.rms_floor != config.feature.rms_floor:
        raise ValueError("feature contract RMS floor differs from the configuration")
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        voiced_crop_probability=config.training.voiced_crop_probability,
    )
    if len(dataset.speaker_to_id) != config.num_speakers:
        raise ValueError("configured speaker count differs from training manifest")
    sampler = HierarchicalBatchSampler(entries, config)
    sampling_audit = sampler.sampling_audit(config.training.max_steps)
    sampling_audit_sha256 = json_sha256(sampling_audit)
    loader_generator = torch.Generator().manual_seed(config.sampling.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_features,
        num_workers=config.sampling.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=config.sampling.prefetch_factor,
        persistent_workers=config.sampling.persistent_workers,
        generator=loader_generator,
    )
    model = HARPCore(
        config.model, config.harmonic, feature_contract, config.num_speakers
    ).to(device)
    runtime.update(
        configure_heavy_linears(model, config.model.heavy_linear_precision)
    )
    system = HARPFlow(
        model,
        transform,
        speaker_drop_probability=config.training.speaker_drop_probability,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    )
    optimizer = build_optimizer(model, config.optimizer, fused=device.type == "cuda")
    base_rates = [group["lr"] for group in optimizer.param_groups]
    contract = checkpoint_contract(
        config,
        model,
        transform_sha256=transform_sha256,
        transform_audit_sha256=transform_audit_sha256,
        feature_contract_sha256=feature_contract_sha256,
        manifest_sha256=manifest_hash,
        exposure_semantics_sha256=exposure_hash,
        batch_runtime_sha256=batch_runtime_hash(config),
        sampling_audit_sha256=sampling_audit_sha256,
        runtime=runtime,
        source_control=source_control,
    )
    if not args.no_compile:
        compile_model_in_place(
            model,
            config.training.compile_mode,
            epilogue_fusion=config.training.inductor_epilogue_fusion,
            shape_padding=config.training.inductor_shape_padding,
        )
        runtime["compiled"] = True
    else:
        runtime["compiled"] = False
    progress, epoch, batch_offset = TrainingProgress(), 0, 0
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    run_id = _prepare_output_directory(
        args.output,
        args.resume,
        contract,
        config,
        exposure_semantics(config, manifest_hash),
        batch_runtime(config),
        source_control,
    )
    _write_immutable_json(args.output / "sampling-audit.json", sampling_audit)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_type") != "full":
            raise ValueError("training can resume only from a full checkpoint")
        validate_checkpoint_contract(payload["contract"], config)
        validate_contract_identity(payload["contract"], contract)
        if payload.get("run_id") != run_id:
            raise ValueError("resume checkpoint belongs to another run")
        if payload["speaker_to_id"] != dataset.speaker_to_id:
            raise ValueError("resume speaker mapping differs")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        ema = {name: value.to(device) for name, value in payload["ema"].items()}
        progress = TrainingProgress.from_dict(payload["progress"])
        epoch = int(payload["epoch"])
        batch_offset = int(payload["batch_offset"])
        torch.set_rng_state(payload["torch_rng_state"])
        if payload["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
    target_steps = args.steps or config.training.max_steps
    if target_steps > config.training.max_steps or target_steps <= progress.global_step:
        raise ValueError("invalid target step")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logged_frames = 0
    started = time.perf_counter()
    while progress.global_step < target_steps:
        sampler.set_epoch(epoch, batch_offset)
        for batch_index, batch in enumerate(loader, start=batch_offset):
            counts = progress.batch_counts(batch)
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.items()
            }
            scale = warmup_scale(
                progress,
                counts["valid_frames"],
                config.training.warmup_valid_frames,
            )
            for group, base_rate in zip(
                optimizer.param_groups, base_rates, strict=True
            ):
                group["lr"] = base_rate * scale
            telemetry_due = (
                progress.global_step + 1
            ) % config.training.telemetry_every_steps == 0
            system.capture_model_inputs = telemetry_due
            loss = system(batch)
            _require_finite(loss.total, "loss", progress.global_step)
            loss.total.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.grad_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            clip_coefficient = min(
                1.0,
                config.training.grad_clip_norm / max(float(grad_norm), 1e-12),
            )
            telemetry = None
            activations = None
            if telemetry_due:
                telemetry = optimizer_role_telemetry(
                    model, optimizer, clip_coefficient=clip_coefficient
                )
                if system.last_model_inputs is None:
                    raise RuntimeError("activation telemetry inputs were not captured")
                activations = model_activation_telemetry(
                    model, system.last_model_inputs
                )
                system.last_model_inputs = None
            optimizer.zero_grad(set_to_none=True)
            ema_decay = ema_decay_for_batch(config.training, counts["valid_frames"])
            _update_ema(ema, model, ema_decay)
            previous_valid_frames = progress.seen_valid_frames
            progress.advance(counts, clipped=clip_coefficient < 1.0)
            batch_offset = batch_index + 1
            logged_frames += counts["valid_frames"]
            if (
                progress.global_step % config.training.log_every_steps == 0
                or progress.global_step == target_steps
            ):
                elapsed = time.perf_counter() - started
                event = {
                    "step": progress.global_step,
                    "loss": float(loss.total.detach()),
                    "grad_norm": float(grad_norm),
                    "clip_coefficient": clip_coefficient,
                    "clip_hit_rate": progress.clipped_updates / progress.global_step,
                    "warmup_scale": scale,
                    "ema_step_decay": ema_decay,
                    "ema_half_life_valid_frames": ema_half_life_valid_frames(
                        config.training
                    ),
                    **progress.to_dict(),
                    "branch_mix_fraction": model.branch_mix_fractions(),
                    "frames_per_second": logged_frames / elapsed,
                    "lambda_floor_fraction": loss.lambda_floor_fraction,
                    "q_floor_fraction": loss.q_floor_fraction,
                    **_device_memory(device),
                }
                if telemetry is not None:
                    event["optimizer_role_telemetry"] = telemetry
                    event["model_activation_telemetry"] = activations
                print(json.dumps(event), flush=True)
                _append_run_event(args.output, "train", event)
                started, logged_frames = time.perf_counter(), 0
            milestones = _crossed_milestones(
                config, previous_valid_frames, progress.seen_valid_frames
            )
            if milestones:
                _append_run_event(
                    args.output,
                    "frame_milestone",
                    {"progress": progress.to_dict(), "milestones": milestones},
                )
            full_due = (
                cadence_crossed(
                    previous_valid_frames,
                    progress.seen_valid_frames,
                    config.training.full_checkpoint_every_valid_frames,
                )
                or progress.global_step == target_steps
            )
            audit_due = cadence_crossed(
                previous_valid_frames,
                progress.seen_valid_frames,
                config.training.audit_checkpoint_every_valid_frames,
            ) or any(name in {"local", "endpoint", "full_panel"} for name in milestones)
            if full_due or audit_due:
                kind = "full" if full_due else "audit"
                path = args.output / (
                    f"{kind}-step-{progress.global_step:07d}-"
                    f"frames-{progress.seen_valid_frames:012d}.pt"
                )
                _save_checkpoint(
                    path,
                    kind,
                    run_id,
                    contract,
                    config,
                    transform,
                    feature_contract,
                    model,
                    optimizer,
                    ema,
                    progress,
                    epoch,
                    batch_offset,
                    dataset.speaker_to_id,
                )
                _index_checkpoint(args.output, path, kind, progress, contract)
                requested_audits = [
                    name
                    for name in milestones
                    if name in {"local", "endpoint", "full_panel"}
                ]
                if requested_audits:
                    _append_audit_request(
                        args.output,
                        path,
                        requested_audits,
                        progress,
                        contract,
                    )
            if progress.global_step >= target_steps:
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
    kind: str,
    run_id: str,
    contract: dict,
    config: HARPConfig,
    transform: FlowTransform,
    feature_contract: FeatureContract,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, torch.Tensor],
    progress: TrainingProgress,
    epoch: int,
    batch_offset: int,
    speaker_to_id: dict[str, int],
) -> None:
    if kind not in {"audit", "full"}:
        raise ValueError("checkpoint kind must be audit or full")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        payload = {
            "checkpoint_type": kind,
            "run_id": run_id,
            "contract": contract,
            "config": config.to_dict(),
            "progress": progress.to_dict(),
            "model": model.state_dict(),
            "ema": ema,
            "speaker_to_id": speaker_to_id,
        }
        if kind == "full":
            payload.update(
                {
                    "epoch": epoch,
                    "batch_offset": batch_offset,
                    "optimizer": optimizer.state_dict(),
                    "flow_transform": transform.to_payload(),
                    "feature_contract": feature_contract.to_payload(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                }
            )
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _require_finite(value: torch.Tensor, name: str, step: int) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"non-finite {name} at step {step}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_output_directory(
    output: Path,
    resume: Path | None,
    contract: dict,
    config: HARPConfig,
    exposure: dict,
    batch: dict,
    git: dict,
) -> str:
    run_path = output / "run.json"
    if resume is None:
        if output.exists():
            raise FileExistsError("new training output directory must not exist")
        output.mkdir(parents=True)
        run_id = f"harp-{int(time.time_ns())}"
        run = {
            "run_id": run_id,
            "contract": contract,
            "git": git,
            "resolved_config": config.to_dict(),
            "exposure_semantics": exposure,
            "batch_runtime": batch,
        }
        run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
        _append_run_event(output, "run_created", run)
        return run_id
    if not output.is_dir() or not run_path.is_file():
        raise FileNotFoundError("resume requires the existing run directory")
    run = json.loads(run_path.read_text())
    validate_contract_identity(run.get("contract", {}), contract)
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("resume run metadata has no run_id")
    _append_run_event(
        output,
        "resume_started",
        {"checkpoint": str(resume.resolve()), "runtime": contract["runtime"]},
    )
    return run_id


def _load_transform_audit(
    path: Path, transform_sha256: str, exposure_sha256: str
) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("contract_accepted") is not True:
        raise ValueError("flow transform independent audit was not accepted")
    if payload.get("flow_transform_sha256") != transform_sha256:
        raise ValueError("flow transform audit refers to another transform")
    if payload.get("exposure_semantics_hash") != exposure_sha256:
        raise ValueError("flow transform audit exposure semantics differ")
    return payload


def _runtime_metadata() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "cudnn": torch.backends.cudnn.version(),
        "triton": _package_version("triton"),
    }


def _device_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"allocated_gib": 0.0, "reserved_gib": 0.0}
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _git_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", "-C", str(root), *arguments), text=True
        ).strip()

    status = git("status", "--porcelain=v1")
    diff = subprocess.check_output(("git", "-C", str(root), "diff", "--binary"))
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _append_run_event(output: Path, event_type: str, payload: dict) -> None:
    event = {
        "type": event_type,
        "time_unix": time.time(),
        **payload,
    }
    with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_immutable_json(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise ValueError(f"immutable run artifact differs: {path.name}")
        return
    path.write_text(serialized)


def _index_checkpoint(
    output: Path,
    path: Path,
    kind: str,
    progress: TrainingProgress,
    contract: dict,
) -> None:
    _append_run_event(
        output,
        "checkpoint",
        {
            "checkpoint_type": kind,
            "filename": path.name,
            "sha256": _sha256(path),
            "progress": progress.to_dict(),
            "config_sha256": contract["config_sha256"],
        },
    )
    with (output / "checkpoint_index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "step": progress.global_step,
                    "seen_valid_frames": progress.seen_valid_frames,
                    "type": kind,
                    "filename": path.name,
                    "sha256": _sha256(path),
                    "config_sha256": contract["config_sha256"],
                    "time_unix": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _append_audit_request(
    output: Path,
    checkpoint: Path,
    audits: list[str],
    progress: TrainingProgress,
    contract: dict,
) -> None:
    request = {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "audits": audits,
        "step": progress.global_step,
        "seen_valid_frames": progress.seen_valid_frames,
        "config_sha256": contract["config_sha256"],
        "time_unix": time.time(),
    }
    with (output / "audit_requests.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(request, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _crossed_milestones(
    config: HARPConfig, previous_frames: int, current_frames: int
) -> list[str]:
    cadences = {
        "health": config.training.health_every_valid_frames,
        "local": config.training.local_audit_every_valid_frames,
        "endpoint": config.training.endpoint_every_valid_frames,
        "full_panel": config.training.full_panel_every_valid_frames,
    }
    return [
        name
        for name, interval in cadences.items()
        if cadence_crossed(previous_frames, current_frames, interval)
    ]


if __name__ == "__main__":
    main()
