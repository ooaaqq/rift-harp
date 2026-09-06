"""Plot speaker-code movement over singer adaptation checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor


def code_movement(code: Tensor, previous: Tensor, initial: Tensor) -> dict[str, float]:
    code = code.detach().float().flatten()
    previous = previous.detach().float().flatten()
    initial = initial.detach().float().flatten()
    if code.shape != previous.shape or code.shape != initial.shape:
        raise ValueError("speaker codes must have identical shapes")
    return {
        "distance_from_initial": torch.linalg.vector_norm(code - initial).item(),
        "distance_from_previous": torch.linalg.vector_norm(code - previous).item(),
        "cosine_with_previous": torch.nn.functional.cosine_similarity(
            code, previous, dim=0
        ).item(),
    }


def collect_trajectory(parent_path: Path, checkpoint_dir: Path) -> dict[str, object]:
    parent = torch.load(
        parent_path, map_location="cpu", weights_only=False, mmap=True
    )
    speaker_table = parent["ema"]["speaker.weight"].detach().float()
    initial = speaker_table[:-1].mean(dim=0)

    loaded = []
    for path in checkpoint_dir.glob("adapter-a-*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_type") != "singer_adapter":
            continue
        loaded.append((int(payload["seen_target_valid_frames"]), path, payload))
    loaded.sort(key=lambda item: item[0])
    if not loaded:
        raise ValueError(f"no Stage A checkpoints found in {checkpoint_dir}")

    previous = {"raw": initial, "ema": initial}
    points = []
    for frames, path, payload in loaded:
        point: dict[str, object] = {
            "checkpoint": path.name,
            "step": int(payload["step"]),
            "seen_target_valid_frames": frames,
        }
        for kind, code in (
            ("raw", payload["adapter"]["code"]),
            ("ema", payload["ema"]["code"]),
        ):
            point[kind] = code_movement(code, previous[kind], initial)
            previous[kind] = code.detach().float()
        points.append(point)

    return {
        "artifact_type": "rift_harp_speaker_code_trajectory_v1",
        "parent_checkpoint": str(parent_path),
        "checkpoint_directory": str(checkpoint_dir),
        "initial_code_definition": (
            "mean of parent EMA real-speaker rows, excluding null"
        ),
        "points": points,
    }


def plot_trajectory(trajectory: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    points = trajectory["points"]
    frames = [point["seen_target_valid_frames"] / 1_000_000 for point in points]
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fields = (
        ("distance_from_initial", r"$\|e_t-e_{init}\|_2$"),
        ("distance_from_previous", r"$\|e_t-e_{prev}\|_2$"),
        ("cosine_with_previous", r"$\cos(e_t,e_{prev})$"),
    )
    for axis, (field, label) in zip(axes, fields, strict=True):
        for kind, style in (("raw", "o-"), ("ema", "s--")):
            axis.plot(
                frames,
                [point[kind][field] for point in points],
                style,
                label=kind.upper(),
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Target valid frames (millions)")
    figure.suptitle("Stage A speaker-code movement")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or args.checkpoint_dir / "speaker-code-trajectory"
    trajectory = collect_trajectory(args.parent, args.checkpoint_dir)
    json_path = prefix.with_suffix(".json")
    png_path = prefix.with_suffix(".png")
    json_path.write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")
    plot_trajectory(trajectory, png_path)
    print(json.dumps({"json": str(json_path), "plot": str(png_path)}))


if __name__ == "__main__":
    main()
