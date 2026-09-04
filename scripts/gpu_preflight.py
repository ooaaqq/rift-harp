from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path

import torch

from harp.config import HARPConfig
from harp.feature_contract import FeatureContract, validate_feature_contract
from harp.flow import HARPFlow
from harp.flow_transform import FlowTransform
from harp.model import HARPCore
from harp.optimizer import build_optimizer
from harp.performance import compile_model_in_place, configure_cuda
from harp.precision import configure_heavy_linears
from harp.training_state import ema_decay_for_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Full HARP RTX preflight")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--rotation-steps", type=int, default=30)
    parser.add_argument(
        "--stage-one-only",
        action="store_true",
        help="validate repeated 96x256 steps before warming the other shapes",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = HARPConfig.load(args.config)
    device = torch.device("cuda")
    runtime = configure_cuda(
        device,
        sdpa_backend=config.training.sdpa_backend,
        allow_tf32=config.training.allow_tf32,
    )
    transform = FlowTransform.load(config.flow.transform_path).to(device)
    feature_contract = FeatureContract.load(config.feature.contract_path)
    validate_feature_contract(
        feature_contract, config.model, config.harmonic, config.feature
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
    optimizer = build_optimizer(model, config.optimizer, fused=True)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    shapes = [
        (
            min(
                config.sampling.batch_size,
                config.sampling.batch_frame_budget // frames,
            ),
            frames,
        )
        for frames in config.training.frame_buckets
    ]
    if shapes != [(96, 256), (64, 384), (48, 512)]:
        raise ValueError(f"unexpected canonical shapes: {shapes}")
    tested_shapes = shapes[:1] if args.stage_one_only else shapes

    eager_batch = _batch(config, *shapes[0], device)
    _step(system, model, optimizer, eager_batch)
    _update_ema(
        ema,
        model,
        ema_decay_for_batch(config.training, int(eager_batch["mask"].sum())),
    )
    dtypes = _dtype_report(model, optimizer, ema)
    if any(values != ["torch.float32"] for values in dtypes.values()):
        raise ValueError(f"preflight precision contract failed: {dtypes}")

    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()
    compile_model_in_place(
        model,
        config.training.compile_mode,
        epilogue_fusion=config.training.inductor_epilogue_fusion,
        shape_padding=config.training.inductor_shape_padding,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    monitor = NvidiaMemoryMonitor()
    monitor.start()
    timings: dict[str, list[float]] = {
        f"{batch}x{frames}": [] for batch, frames in tested_shapes
    }
    shape_memory = {}
    try:
        for batch_size, frames in tested_shapes:
            batch = _batch(config, batch_size, frames, device)
            _timed_step(
                system,
                model,
                optimizer,
                ema,
                batch,
                config,
                timings[f"{batch_size}x{frames}"],
            )
            shape_memory[f"{batch_size}x{frames}"] = {
                "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
            }
        order = tested_shapes * (
            (args.rotation_steps + len(tested_shapes) - 1) // len(tested_shapes)
        )
        random.Random(2026).shuffle(order)
        for batch_size, frames in order[: args.rotation_steps]:
            batch = _batch(config, batch_size, frames, device)
            _timed_step(
                system,
                model,
                optimizer,
                ema,
                batch,
                config,
                timings[f"{batch_size}x{frames}"],
            )
    finally:
        monitor.stop()
    torch.cuda.synchronize(device)
    unique_graphs = int(counters["stats"]["unique_graphs"])
    unexpected_graphs = max(0, unique_graphs - len(tested_shapes))

    report = {
        **runtime,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "canonical_shapes": shapes,
        "tested_shapes": tested_shapes,
        "cuda_graphs_enabled": True,
        "dtype_contract": dtypes,
        "unique_compiled_graphs": unique_graphs,
        "unexpected_recompile_graphs": unexpected_graphs,
        "compile_counters": {
            category: dict(values)
            for category, values in counters.items()
            if category in {"frames", "stats", "inductor", "aot_autograd"}
        },
        "memory_before_shape_warm_gib": {
            "allocated": baseline_allocated / 2**30,
            "reserved": baseline_reserved / 2**30,
        },
        "memory_after_each_shape_warm_gib": shape_memory,
        "compile_reserved_delta_gib": (
            max(value["reserved_gib"] for value in shape_memory.values())
            - baseline_reserved / 2**30
        ),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "nvidia_process_peak_mib": monitor.peak_mib,
        "bucket": {
            name: {
                "steps": len(values),
                "seconds_per_step_median": float(torch.tensor(values).median()),
                "requested_frames_per_second": int(name.split("x")[0])
                * int(name.split("x")[1])
                / float(torch.tensor(values).median()),
            }
            for name, values in timings.items()
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if unexpected_graphs:
        raise RuntimeError(f"unexpected compiled graphs detected: {unique_graphs}")


def _batch(
    config: HARPConfig, batch_size: int, frames: int, device: torch.device
) -> dict[str, torch.Tensor]:
    mask = torch.ones(batch_size, frames, dtype=torch.bool, device=device)
    return {
        "mel": torch.randn(
            batch_size, frames, config.model.mel_channels, device=device
        ),
        "content": torch.randn(
            batch_size, frames, config.model.content_dim, device=device
        ),
        "f0": torch.rand(batch_size, frames, 1, device=device) * 900 + 60,
        "rms": torch.pow(
            10.0, torch.rand(batch_size, frames, 1, device=device) * 4 - 5
        ),
        "speaker": torch.randint(config.num_speakers, (batch_size,), device=device),
        "mask": mask,
    }


def _step(
    system: HARPFlow,
    model: HARPCore,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = system(batch).total
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite preflight loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()


def _timed_step(
    system: HARPFlow,
    model: HARPCore,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: HARPConfig,
    timings: list[float],
) -> None:
    torch.cuda.synchronize()
    started = time.perf_counter()
    _step(system, model, optimizer, batch)
    _update_ema(
        ema,
        model,
        ema_decay_for_batch(config.training, int(batch["mask"].sum())),
    )
    torch.cuda.synchronize()
    timings.append(time.perf_counter() - started)


@torch.no_grad()
def _update_ema(ema: dict[str, torch.Tensor], model: HARPCore, decay: float) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema[name].lerp_(value, 1 - decay)
        else:
            ema[name].copy_(value)


def _dtype_report(
    model: HARPCore,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    states = list(optimizer.state.values())
    return {
        "parameters": sorted({str(value.dtype) for value in model.parameters()}),
        "ema": sorted(
            {str(value.dtype) for value in ema.values() if value.is_floating_point()}
        ),
        "adam_exp_avg": sorted({str(value["exp_avg"].dtype) for value in states}),
        "adam_exp_avg_sq": sorted({str(value["exp_avg_sq"].dtype) for value in states}),
    }


class NvidiaMemoryMonitor:
    def __init__(self) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            try:
                output = subprocess.check_output(
                    (
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ),
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                values = []
                for line in output.splitlines():
                    process, memory = (part.strip() for part in line.split(",", 1))
                    if int(process) == os.getpid():
                        values.append(int(memory))
                self.peak_mib = max([self.peak_mib, *values])
            except (OSError, subprocess.SubprocessError, ValueError):
                return


if __name__ == "__main__":
    main()
