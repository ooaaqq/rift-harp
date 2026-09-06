from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from .checkpoint import validate_checkpoint_contract, validate_external_artifacts
from .config import HARPConfig
from .data import FeatureDataset, SampleRequest, collate_features
from .feature_contract import FeatureContract
from .flow import HARPFlow
from .flow_transform import FlowTransform
from .manifest import load_manifest, manifest_sha256
from .model import HARPCore
from .performance import compile_model_in_place, configure_cuda
from .precision import configure_heavy_linears


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-anchor HARP context sweep")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=(64, 128, 192, 256, 384, 512, 768)
    )
    parser.add_argument("--score-frames", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    contexts = tuple(sorted(set(args.contexts)))
    if not contexts or min(contexts) <= 0 or max(contexts) > 768:
        raise ValueError("contexts must be in 1..768")
    if (
        args.score_frames <= 0
        or args.score_frames > min(contexts)
        or any((768 - context) % 2 for context in contexts)
    ):
        raise ValueError("contexts must permit centered windows")

    config = HARPConfig.load(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_contract(checkpoint["contract"], config)
    validate_external_artifacts(
        checkpoint["contract"],
        transform_path=config.flow.transform_path,
        transform_audit_path=config.flow.audit_path,
        feature_contract_path=config.feature.contract_path,
        manifest_path=args.manifest,
    )
    panel = json.loads(args.panel_lock.read_text(encoding="utf-8"))
    entries = load_manifest(args.manifest)
    id_to_index = {entry.id: index for index, entry in enumerate(entries)}
    samples = sorted(panel["samples"], key=lambda item: int(item["ordinal"]))
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        speaker_to_id=checkpoint["speaker_to_id"],
        voiced_crop_probability=config.training.voiced_crop_probability,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        configure_cuda(
            device,
            sdpa_backend=config.training.sdpa_backend,
            allow_tf32=config.training.allow_tf32,
        )
    transform = FlowTransform.load(config.flow.transform_path).to(device)
    feature_contract = FeatureContract.load(config.feature.contract_path)
    model = HARPCore(
        config.model, config.harmonic, feature_contract, config.num_speakers
    ).to(device)
    configure_heavy_linears(model, config.model.heavy_linear_precision)
    if device.type == "cuda":
        compile_model_in_place(model, "default")
    model.load_state_dict(checkpoint["ema"], strict=True)
    system = HARPFlow(
        model,
        transform,
        speaker_drop_probability=0,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    ).eval()
    generator = torch.Generator(device="cpu").manual_seed(
        int(panel["protocol"]["seed"])
    )
    noise = torch.randn(
        len(samples), 768, config.model.mel_channels, generator=generator
    )
    result = {
        "artifact_type": "harp_fixed_anchor_context_sweep_v1",
        "manifest_sha256": manifest_sha256(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_progress": checkpoint["progress"],
        "panel_lock": str(args.panel_lock),
        "protocol": {
            "mother_frames": 768,
            "contexts": list(contexts),
            "score_frames": args.score_frames,
            "score_region": "center of 768 mother crop",
            "same_noise_prefix": True,
            "same_speaker_target_frontend": True,
            "solver": "euler",
            "steps": args.steps,
        },
        "models": {"harp_ema_correct": {}},
    }
    with torch.inference_mode():
        for context in contexts:
            left = (768 - context) // 2
            score_left = (context - args.score_frames) // 2
            rows = []
            for offset in range(0, len(samples), args.batch_size):
                items = samples[offset : offset + args.batch_size]
                loaded = [
                    dataset[
                        SampleRequest(
                            id_to_index[str(item["entry_id"])],
                            768,
                            int(item["ordinal"]),
                            int(item["start_frame"]),
                        )
                    ]
                    for item in items
                ]
                batch_cpu = collate_features(loaded)
                batch = {
                    name: (
                        value[:, left : left + context] if value.ndim >= 2 else value
                    ).to(device)
                    for name, value in batch_cpu.items()
                    if name in {"content", "f0", "rms", "speaker", "mask"}
                }
                prediction = system.sample(
                    batch["content"],
                    batch["f0"],
                    batch["rms"],
                    batch["speaker"],
                    batch["mask"],
                    steps=args.steps,
                    method="euler",
                    guidance_strength=1.0,
                    initial_noise=noise[
                        offset : offset + len(items), left : left + context
                    ].to(device),
                ).cpu()
                target = batch_cpu["mel"][
                    :, left + score_left : left + score_left + args.score_frames
                ]
                rms = batch_cpu["rms"][
                    :, left + score_left : left + score_left + args.score_frames
                ]
                mask = batch_cpu["mask"][
                    :, left + score_left : left + score_left + args.score_frames
                ].bool()
                active = mask & (rms.squeeze(-1) > 0.001)
                error = (
                    (
                        prediction[
                            :, score_left : score_left + args.score_frames
                        ].float()
                        - target.float()
                    )
                    .square()
                    .mean(-1)
                )
                for index, item in enumerate(items):
                    selected = error[index][active[index]]
                    rows.append(
                        {
                            "ordinal": int(item["ordinal"]),
                            "entry_id": str(item["entry_id"]),
                            "active_mse": float(selected.mean())
                            if selected.numel()
                            else None,
                            "active_frames": int(selected.numel()),
                        }
                    )
            values = [
                row["active_mse"] for row in rows if row["active_mse"] is not None
            ]
            result["models"]["harp_ema_correct"][str(context)] = {
                "samples": rows,
                "mean_active_mse": statistics.fmean(values),
                "median_active_mse": statistics.median(values),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result["models"]["harp_ema_correct"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
