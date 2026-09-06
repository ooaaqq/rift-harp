from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from .checkpoint import validate_checkpoint_contract, validate_external_artifacts
from .config import HARPConfig
from .data import FeatureDataset, SampleRequest, collate_features
from .feature_contract import FeatureContract
from .flow import HARPFlow
from .flow_transform import FlowTransform, orthonormal_dct
from .local_audit_cli import BANDS, MetricAccumulator
from .manifest import load_manifest, manifest_sha256
from .model import HARPCore
from .panel_cli import validate_panel_features
from .performance import configure_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description="HARP raw/EMA endpoint audit")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--method", choices=("euler", "heun"), default="euler")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
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
    panel_artifact = json.loads(args.panels.read_text())
    if panel_artifact.get("manifest_sha256") != manifest_sha256(args.manifest):
        raise ValueError("fixed panel manifest hash differs")
    entries = load_manifest(args.manifest)
    id_to_index = {entry.id: index for index, entry in enumerate(entries)}
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        speaker_to_id=checkpoint["speaker_to_id"],
        voiced_crop_probability=config.training.voiced_crop_probability,
    )
    batches = _load_batches(dataset, id_to_index, panel_artifact)
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
    system = HARPFlow(
        model,
        transform,
        speaker_drop_probability=0,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    ).eval()
    dct = orthonormal_dct(config.model.mel_channels).to(device)
    accumulators: dict[
        tuple[str, str, str, int | None, str, str], MetricAccumulator
    ] = defaultdict(MetricAccumulator)
    with torch.inference_mode():
        for state_name in ("raw", "ema"):
            model.load_state_dict(
                checkpoint["model" if state_name == "raw" else "ema"], strict=True
            )
            model.eval()
            for (panel_name, requested_frames), cpu_batch in batches.items():
                batch = {name: value.to(device) for name, value in cpu_batch.items()}
                initial_noise = _fixed_noise(batch, config.model.mel_channels, device)
                speaker_conditions = {
                    "correct": batch["speaker"],
                    "null": torch.full_like(batch["speaker"], model.null_speaker_id),
                    "wrong": (batch["speaker"] + 1) % config.num_speakers,
                }
                target_dct = batch["mel"].float() @ dct.T
                voiced = batch["mask"] & (batch["f0"][..., 0] > 0)
                strata = {
                    "all": batch["mask"],
                    "voiced": voiced,
                    "unvoiced": batch["mask"] & ~voiced,
                }
                for condition_name, speaker in speaker_conditions.items():
                    prediction = system.sample(
                        batch["content"],
                        batch["f0"],
                        batch["rms"],
                        speaker,
                        batch["mask"],
                        steps=args.steps,
                        guidance_strength=1.0,
                        method=args.method,
                        initial_noise=initial_noise,
                    )
                    prediction_dct = prediction @ dct.T
                    for stratum, selection in strata.items():
                        for band, (start, end) in BANDS.items():
                            prediction_band = prediction_dct[..., start:end][selection]
                            target_band = target_dct[..., start:end][selection]
                            for length_key in (None, requested_frames):
                                accumulators[
                                    (
                                        state_name,
                                        panel_name,
                                        condition_name,
                                        length_key,
                                        stratum,
                                        band,
                                    )
                                ].update(prediction_band, target_band)
    payload = {
        "artifact_type": "harp_endpoint_audit_v2",
        "checkpoint": str(args.checkpoint),
        "checkpoint_progress": checkpoint["progress"],
        "panel_artifact": str(args.panels),
        "solver": {"method": args.method, "steps": args.steps},
        "states": ["raw", "ema"],
        "speaker_conditions": ["correct", "null", "wrong"],
        "results": [
            {
                "state": key[0],
                "panel": key[1],
                "speaker_condition": key[2],
                "requested_frames": key[3],
                "stratum": key[4],
                "band": key[5],
                **value.report(),
            }
            for key, value in sorted(
                accumulators.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    item[0][2],
                    -1 if item[0][3] is None else item[0][3],
                    item[0][4],
                    item[0][5],
                ),
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_batches(
    dataset: FeatureDataset,
    id_to_index: dict[str, int],
    artifact: dict,
) -> dict[tuple[str, int], dict[str, Tensor]]:
    grouped: dict[tuple[str, int], list[dict[str, Tensor]]] = defaultdict(list)
    for panel_name, items in artifact["panels"].items():
        for item in items:
            entry = dataset.entries[id_to_index[item["entry_id"]]]
            validate_panel_features(entry, item)
            sample = dataset[
                SampleRequest(
                    id_to_index[item["entry_id"]],
                    int(item["requested_frames"]),
                    int(item["noise_seed"]),
                    int(item["crop_start"]),
                )
            ]
            sample["noise_seed"] = torch.tensor(int(item["noise_seed"]))
            grouped[(panel_name, int(item["requested_frames"]))].append(sample)
    return {key: collate_features(samples) for key, samples in grouped.items()}


def _fixed_noise(
    batch: dict[str, Tensor], channels: int, device: torch.device
) -> Tensor:
    parts = []
    for seed, frames in zip(
        batch["noise_seed"], batch["requested_length"], strict=True
    ):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        parts.append(
            torch.randn(int(frames), channels, device=device, generator=generator)
        )
    return torch.stack(parts)


if __name__ == "__main__":
    main()
