"""Speaker-only post-training for a frozen HARP foundation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Sampler

from .checkpoint import validate_checkpoint_contract
from .config import HARPConfig
from .data import FeatureDataset, SampleRequest, collate_features
from .feature_contract import FeatureContract, validate_feature_contract
from .flow import HARPFlow
from .flow_transform import FlowTransform
from .manifest import load_manifest, manifest_sha256
from .model import HARPCore
from .precision import configure_heavy_linears
from .training_state import warmup_scale


class SingerAdapter(nn.Module):
    """The only trainable state for a target singer.

    Offsets are indexed by the 16 block modulations followed by final
    modulation.  They are added after each frozen speaker projection.
    """

    def __init__(self, model: HARPCore, *, initial_code: Tensor | None = None) -> None:
        super().__init__()
        if initial_code is None:
            initial_code = model.speaker.weight[:-1].detach().mean(dim=0)
        if initial_code.shape != (model.config.speaker_code_dim,):
            raise ValueError("target speaker code has an invalid shape")
        self.code = nn.Parameter(initial_code.float().clone())
        self.offsets = nn.Parameter(
            torch.zeros(model.config.depth + 1, model.config.adaln_rank)
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def stage(self, name: str) -> None:
        if name not in {"a", "b"}:
            raise ValueError("adaptation stage must be 'a' or 'b'")
        self.code.requires_grad_(name == "a")
        self.offsets.requires_grad_(name == "b")

    def payload(self) -> dict[str, Tensor | str]:
        return {
            "stage": "adapter",
            "code": self.code.detach().cpu(),
            "offsets": self.offsets.detach().cpu(),
        }

    def load_payload(self, payload: dict[str, object]) -> None:
        code = torch.as_tensor(payload["code"], dtype=torch.float32)
        offsets = torch.as_tensor(payload["offsets"], dtype=torch.float32)
        if code.shape != self.code.shape or offsets.shape != self.offsets.shape:
            raise ValueError("adapter payload shape differs from this foundation")
        self.code.data.copy_(code)
        self.offsets.data.copy_(offsets)


class AdaptationBatchSampler(Sampler[list[SampleRequest]]):
    """Song/recording-aware enough for a single singer, with fixed shapes."""

    def __init__(self, entries: list, config: HARPConfig, *, seed: int) -> None:
        self.entries = [
            entry
            for entry in entries
            if entry.split == "train" and entry.quality_status == "accepted"
        ]
        if not self.entries:
            raise ValueError("adaptation manifest has no accepted train recordings")
        self.buckets = config.training.frame_buckets
        self.probabilities = config.training.bucket_probabilities
        self.batch_budget = 12_288
        self.steps_per_epoch = max(1, config.sampling.steps_per_epoch)
        self.seed = seed
        self.epoch = 0
        self.start_step = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        self.epoch, self.start_step = epoch, start_step

    def __iter__(self):
        for step in range(self.start_step, self.steps_per_epoch):
            rng = random.Random(self.seed + self.epoch * 1_000_003 + step * 97_003)
            frames = rng.choices(self.buckets, self.probabilities, k=1)[0]
            batch_size = self.batch_budget // frames
            requests = []
            for _position in range(batch_size):
                index = rng.randrange(len(self.entries))
                requests.append(SampleRequest(index, frames, rng.randrange(2**63)))
            yield requests


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_parent(
    config: HARPConfig,
    checkpoint_path: Path,
    transform: FlowTransform,
    feature: FeatureContract,
    device: torch.device,
) -> tuple[HARPCore, dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_type") not in {"full", "audit"}:
        raise ValueError("parent must be a HARP full or audit checkpoint")
    validate_checkpoint_contract(payload["contract"], config)
    model = HARPCore(config.model, config.harmonic, feature, config.num_speakers)
    configure_heavy_linears(model, config.model.heavy_linear_precision)
    model.load_state_dict(payload["ema"], strict=True)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def adapt(config: HARPConfig, args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    entries = load_manifest(args.manifest)
    train_entries = [
        entry
        for entry in entries
        if entry.split == "train" and entry.quality_status == "accepted"
    ]
    transform = FlowTransform.load(config.flow.transform_path).to(device)
    feature = FeatureContract.load(config.feature.contract_path)
    validate_feature_contract(feature, config.model, config.harmonic, config.feature)
    parent_path = Path(args.parent)
    model, parent = _load_parent(config, parent_path, transform, feature, device)
    adapter = SingerAdapter(model).to(device)
    if args.init_adapter:
        init_payload = torch.load(
            args.init_adapter, map_location="cpu", weights_only=False
        )
        if init_payload.get("checkpoint_type") != "singer_adapter":
            raise ValueError("--init-adapter must be a singer adapter checkpoint")
        adapter.load_payload(init_payload.get("ema", init_payload["adapter"]))
    adapter.stage(args.stage)
    trainable = [
        parameter for parameter in adapter.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
        foreach=False,
    )
    dataset = FeatureDataset(
        train_entries, config.model.mel_channels, config.model.content_dim
    )
    sampler = AdaptationBatchSampler(train_entries, config, seed=args.seed)
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
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("adaptation output directory already exists")
    output.mkdir(parents=True)
    run = {
        "artifact_type": "rift_harp_singer_adaptation_v1",
        "stage": args.stage,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": _sha256(parent_path),
        "parent_progress": parent.get("progress"),
        "manifest_sha256": manifest_sha256(args.manifest),
        "config_sha256": config.digest(),
        "adapter_parameter_count": adapter.parameter_count,
        "created_at_unix": time.time(),
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    ema = {name: value.detach().clone() for name, value in adapter.state_dict().items()}
    target_frames = args.valid_frames
    seen_frames = 0
    step = 0
    model.train()
    adapter.train()
    epoch = 0
    while seen_frames < target_frames:
        sampler.set_epoch(epoch)
        for batch in loader:
            counts = int(batch["length"].sum())
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.items()
            }
            progress = type("Progress", (), {"seen_valid_frames": seen_frames})()
            scale = warmup_scale(progress, counts, args.warmup_frames)
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * scale
            speaker_code = adapter.code.unsqueeze(0).expand(batch["mel"].shape[0], -1)
            loss = system(
                batch,
                speaker_code_override=speaker_code,
                speaker_offsets=adapter.offsets,
            )
            loss.total.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            decay = 2.0 ** (-counts / args.ema_half_life_frames)
            with torch.no_grad():
                for name, value in adapter.state_dict().items():
                    ema[name].lerp_(value, 1 - decay)
            seen_frames += counts
            step += 1
            if step % args.log_every_steps == 0:
                print(
                    json.dumps(
                        {
                            "stage": args.stage,
                            "step": step,
                            "loss": float(loss.total),
                            "seen_target_valid_frames": seen_frames,
                        }
                    ),
                    flush=True,
                )
            if seen_frames >= target_frames:
                break
        epoch += 1
    _save_adaptation(
        output / f"adapter-{args.stage}-step-{step:06d}.pt",
        run,
        adapter,
        ema,
        optimizer,
        step,
        seen_frames,
    )


def _save_adaptation(
    path: Path,
    run: dict,
    adapter: SingerAdapter,
    ema: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    step: int,
    seen_frames: int,
) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        torch.save(
            {
                "checkpoint_type": "singer_adapter",
                "run": run,
                "adapter": adapter.payload(),
                "ema": ema,
                "optimizer": optimizer.state_dict(),
                "step": step,
                "seen_target_valid_frames": seen_frames,
            },
            temporary,
        )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt a frozen HARP foundation to one singer"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid-frames", type=int, default=12_000_000)
    parser.add_argument("--warmup-frames", type=int, default=500_000)
    parser.add_argument("--ema-half-life-frames", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every-steps", type=int, default=20)
    args = parser.parse_args()
    adapt(HARPConfig.load(args.config), args)


if __name__ == "__main__":
    main()
