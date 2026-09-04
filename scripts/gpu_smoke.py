from __future__ import annotations

import argparse
import json

import torch

from harp.config import HarmonicConfig, ModelConfig
from harp.flow import HARPFlow
from harp.flow_transform import FlowTransform
from harp.model import HARPCore
from harp.optimizer import build_optimizer
from harp.performance import compile_model_in_place, configure_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description="HARP BF16 CUDA numerical smoke")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    runtime = configure_cuda(device)
    model_config = ModelConfig() if args.full else _tiny_config()
    harmonic_config = HarmonicConfig()
    model = HARPCore(model_config, harmonic_config, num_speakers=3).to(device)
    transform = FlowTransform(
        mean=torch.zeros(model_config.mel_channels),
        basis=torch.eye(model_config.mel_channels),
        gain=torch.ones(model_config.mel_channels),
        lambda_raw=torch.ones(model_config.mel_channels),
        lambda_effective=torch.ones(model_config.mel_channels),
        metadata={"artifact_type": "synthetic_smoke_only"},
    )
    system = HARPFlow(model, transform)
    if args.compile:
        compile_model_in_place(model, "max-autotune")
    optimizer = build_optimizer(model, fused=True)
    batch = _batch(model_config, args.frames, device)
    losses = []
    grad_norms = []
    for _ in range(2):
        loss = system(batch)
        loss.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.total.detach()))
        grad_norms.append(float(grad_norm))
    generated = system.eval().sample(
        batch["content"],
        batch["f0"],
        batch["rms"],
        batch["speaker"],
        batch["mask"],
        steps=2,
        guidance_strength=1.5,
    )
    print(
        json.dumps(
            {
                **runtime,
                "full_model": args.full,
                "compiled": args.compile,
                "parameters": sum(value.numel() for value in model.parameters()),
                "losses": losses,
                "grad_norms": grad_norms,
                "generated_finite": bool(torch.isfinite(generated).all()),
                "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        mel_channels=16,
        content_dim=32,
        dim=64,
        depth=4,
        head_dim=16,
        ff_hidden_dim=128,
        kernel_size=5,
        time_code_dim=32,
        speaker_code_dim=32,
        adaln_rank=8,
        adaln_mixer_dim=16,
        harmonic_dim=16,
        harmonic_injection_blocks=(1, 2, 3),
    )


def _batch(
    config: ModelConfig, frames: int, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "mel": torch.randn(1, frames, config.mel_channels, device=device),
        "content": torch.randn(1, frames, config.content_dim, device=device),
        "f0": torch.rand(1, frames, 1, device=device) * 500 + 80,
        "rms": torch.randn(1, frames, 1, device=device),
        "speaker": torch.tensor([1], device=device),
        "mask": torch.ones(1, frames, device=device, dtype=torch.bool),
    }


if __name__ == "__main__":
    main()
