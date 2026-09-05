from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .checkpoint import validate_checkpoint_contract, validate_external_artifacts
from .config import HARPConfig
from .data import FeatureDataset, SampleRequest, collate_features
from .endpoint_audit_cli import _fixed_noise
from .feature_contract import FeatureContract
from .flow import HARPFlow
from .flow_transform import FlowTransform
from .full_panel_cli import _load_raw_manifest, _reference_crop
from .manifest import load_manifest
from .model import HARPCore
from .performance import compile_model_in_place, configure_cuda
from .precision import configure_heavy_linears
from .vocoder import load_pc_nsf, synthesize_pc_nsf

WAVLM_REPOSITORY = "microsoft/wavlm-base-plus-sv"
WAVLM_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HARP A-to-B speaker progress")
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pc-nsf-checkout", type=Path, required=True)
    parser.add_argument("--pc-nsf-lock", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--method", choices=("euler", "heun"), default="euler")
    parser.add_argument("--wavlm-batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps <= 0 or args.wavlm_batch_size <= 0:
        raise ValueError("solver steps and WavLM batch size must be positive")

    pairs_artifact, calibration, anchors = _load_protocol(
        args.pairs, args.calibration, args.anchors
    )
    pairs = pairs_artifact["pairs"]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        pairs = pairs[: args.limit]

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
    entries = load_manifest(args.manifest)
    id_to_index = {entry.id: index for index, entry in enumerate(entries)}
    raw_manifest = _load_raw_manifest(args.manifest)
    dataset = FeatureDataset(
        entries,
        config.model.mel_channels,
        config.model.content_dim,
        speaker_to_id=checkpoint["speaker_to_id"],
        voiced_crop_probability=config.training.voiced_crop_probability,
    )
    items, batch = _load_conversion_batch(
        pairs, dataset, id_to_index, checkpoint["speaker_to_id"]
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
    system = HARPFlow(
        model,
        transform,
        speaker_drop_probability=0,
        lambda_floor=config.flow.lambda_floor,
        q_floor=config.flow.q_floor,
    ).eval()
    vocoder, vocoder_contract = load_pc_nsf(
        args.pc_nsf_checkout,
        args.vocoder_checkpoint,
        args.pc_nsf_lock,
        config,
        device,
    )
    encoder = WavLMSpeakerEncoder(device, args.wavlm_batch_size)

    if args.output.exists():
        raise FileExistsError(f"speaker audit output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    destination = Path(
        tempfile.mkdtemp(dir=args.output.parent, prefix=".speaker-audit.")
    )
    try:
        reference_metrics = _reference_metrics(
            items,
            calibration,
            anchors,
            raw_manifest,
            encoder,
            config,
        )
        results = _render_conversions(
            destination,
            items,
            batch,
            {"raw": checkpoint["model"], "ema": checkpoint["ema"]},
            model,
            system,
            vocoder,
            encoder,
            reference_metrics,
            config,
            device,
            args.steps,
            args.method,
        )
        payload = {
            "artifact_type": "harp_speaker_progress_v1",
            "created_at_unix": time.time(),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "checkpoint_progress": checkpoint["progress"],
            "protocol": {
                "pairs_sha256": _sha256(args.pairs),
                "calibration_sha256": _sha256(args.calibration),
                "anchors_sha256": _sha256(args.anchors),
                "wavlm_repository": WAVLM_REPOSITORY,
                "wavlm_revision": WAVLM_REVISION,
                "progress_definition": (
                    "(converted_margin-source_ground_truth_margin)/"
                    "(target_ground_truth_margin-source_ground_truth_margin)"
                ),
            },
            "vocoder": {
                "checkout_revision": vocoder_contract.revision,
                "checkpoint_sha256": vocoder_contract.checkpoint_sha256,
            },
            "solver": {"method": args.method, "steps": args.steps},
            "results": results,
            "aggregate": _aggregate(results),
        }
        (destination / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(destination, args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
    except BaseException:
        for path in destination.glob("*"):
            path.unlink(missing_ok=True)
        destination.rmdir()
        raise


class WavLMSpeakerEncoder:
    def __init__(self, device: torch.device, batch_size: int) -> None:
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector
        except ImportError as error:
            raise RuntimeError("speaker audit requires transformers") from error
        self.device = device
        self.batch_size = batch_size
        self.extractor = AutoFeatureExtractor.from_pretrained(
            WAVLM_REPOSITORY, revision=WAVLM_REVISION, local_files_only=True
        )
        self.model = WavLMForXVector.from_pretrained(
            WAVLM_REPOSITORY, revision=WAVLM_REVISION, local_files_only=True
        ).to(device).eval()

    @torch.inference_mode()
    def encode(self, waveforms: list[Tensor], source_rate: int) -> Tensor:
        import torchaudio.functional as audio_functional

        prepared = [
            audio_functional.resample(wave.float().cpu(), source_rate, 16_000)
            if source_rate != 16_000
            else wave.float().cpu()
            for wave in waveforms
        ]
        result = []
        for start in range(0, len(prepared), self.batch_size):
            values = self.extractor(
                [wave.numpy() for wave in prepared[start : start + self.batch_size]],
                sampling_rate=16_000,
                padding=True,
                return_tensors="pt",
            )
            embeddings = self.model(
                **{name: value.to(self.device) for name, value in values.items()}
            ).embeddings
            result.append(F.normalize(embeddings.float(), dim=-1).cpu())
        return torch.cat(result)


def speaker_metrics(
    embedding: Tensor, source_prototype: Tensor, target_prototype: Tensor
) -> dict[str, float]:
    embedding = F.normalize(embedding.float(), dim=-1)
    source = float(embedding @ F.normalize(source_prototype.float(), dim=-1))
    target = float(embedding @ F.normalize(target_prototype.float(), dim=-1))
    return {
        "similarity_to_source": source,
        "similarity_to_target": target,
        "target_margin": target - source,
    }


def normalized_progress(
    converted_margin: float, source_margin: float, target_margin: float
) -> float:
    separation = target_margin - source_margin
    if separation <= 0:
        raise ValueError(
            "normalized speaker progress requires positive anchor separation"
        )
    return (converted_margin - source_margin) / separation


def _load_protocol(
    pairs_path: Path, calibration_path: Path, anchors_path: Path
) -> tuple[dict, dict[int, dict], dict[int, dict]]:
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    if pairs.get("schema_version") != 2 or calibration.get("schema_version") != 1:
        raise ValueError("unsupported speaker panel lock schema")
    if calibration.get("pair_spec_sha256") != _sha256(pairs_path):
        raise ValueError("speaker calibration does not match pair lock")
    fingerprints = {
        pairs.get("manifest_fingerprint_sha256"),
        calibration.get("manifest_fingerprint_sha256"),
    }
    if len(fingerprints) != 1:
        raise ValueError("speaker panel manifest fingerprints differ")
    protocol = anchors.get("protocol", {})
    encoder = protocol.get("speaker_encoder", {})
    if encoder != {"repository": WAVLM_REPOSITORY, "revision": WAVLM_REVISION}:
        raise ValueError("speaker anchor WavLM contract differs")
    expected_definition = (
        "(converted_margin-source_ground_truth_margin)/"
        "(target_ground_truth_margin-source_ground_truth_margin)"
    )
    if protocol.get("progress_definition") != expected_definition:
        raise ValueError("speaker progress definition differs")
    calibration_by_pair = {int(item["pair"]): item for item in calibration["pairs"]}
    anchors_by_pair = {int(item["pair"]): item for item in anchors["anchors"]}
    if set(calibration_by_pair) != set(range(len(pairs["pairs"]))) or set(
        anchors_by_pair
    ) != set(calibration_by_pair):
        raise ValueError("speaker panel pair indices are incomplete")
    return pairs, calibration_by_pair, anchors_by_pair


def _load_conversion_batch(
    pairs: list[dict],
    dataset: FeatureDataset,
    id_to_index: dict[str, int],
    speaker_to_id: dict[str, int],
) -> tuple[list[dict], dict[str, Tensor]]:
    samples = []
    for pair_index, pair in enumerate(pairs):
        source = pair["source"]
        entry = dataset.entries[id_to_index[source["id"]]]
        actual_hashes = _feature_hashes(entry)
        if actual_hashes != source["feature_sha256"]:
            raise ValueError(
                f"conversion source features changed for pair {pair_index}"
            )
        sample = dataset[
            SampleRequest(
                id_to_index[source["id"]],
                int(source["frames"]),
                int(pair["seed"]),
                int(source["start_frame"]),
            )
        ]
        sample["speaker"] = torch.tensor(speaker_to_id[pair["target_speaker"]])
        sample["noise_seed"] = torch.tensor(int(pair["seed"]))
        samples.append(sample)
    return pairs, collate_features(samples)


def _feature_hashes(entry) -> dict[str, str]:
    paths = {
        "content": Path(
            entry.content_feature_path or f"{entry.feature_prefix}.content.pt"
        ),
        "f0": Path(f"{entry.feature_prefix}.f0.pt"),
        "rms": Path(f"{entry.feature_prefix}.rms.pt"),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _reference_metrics(
    pairs: list[dict],
    calibration: dict[int, dict],
    anchors: dict[int, dict],
    manifest: dict[str, dict],
    encoder: WavLMSpeakerEncoder,
    config: HARPConfig,
) -> dict[int, dict]:
    output = {}
    for pair_index, pair in enumerate(pairs):
        specs = [
            pair["source"],
            *pair["source_references"],
            *pair["target_references"],
            calibration[pair_index]["target_ground_truth"],
        ]
        waveforms = [
            _locked_reference(spec, manifest, config) for spec in specs
        ]
        embeddings = encoder.encode(waveforms, config.feature.sample_rate)
        source_prototype = F.normalize(embeddings[1:3].mean(0), dim=0)
        target_prototype = F.normalize(embeddings[3:5].mean(0), dim=0)
        measured_source = speaker_metrics(
            embeddings[0], source_prototype, target_prototype
        )
        measured_target = speaker_metrics(
            embeddings[5], source_prototype, target_prototype
        )
        expected = anchors[pair_index]
        _validate_anchor(measured_source, expected["source_ground_truth"], pair_index)
        _validate_anchor(measured_target, expected["target_ground_truth"], pair_index)
        output[pair_index] = {
            "source_prototype": source_prototype,
            "target_prototype": target_prototype,
            "source_margin": float(expected["source_ground_truth"]["target_margin"]),
            "target_margin": float(expected["target_ground_truth"]["target_margin"]),
            "anchor_separation": float(expected["anchor_separation"]),
        }
    return output


def _locked_reference(
    spec: dict, manifest: dict[str, dict], config: HARPConfig
) -> Tensor:
    source = manifest[spec["id"]]
    path = Path(source["audio_path"])
    if _sha256(path) != spec["audio_sha256"]:
        raise ValueError(f"speaker reference audio changed: {spec['id']}")
    waveform, _ = _reference_crop(
        path,
        int(spec["start_frame"]),
        int(spec["frames"]),
        config.feature.sample_rate,
        config.feature.hop_length,
    )
    return waveform


def _validate_anchor(measured: dict, expected: dict, pair_index: int) -> None:
    if any(abs(measured[name] - float(expected[name])) > 5e-4 for name in measured):
        raise ValueError(f"WavLM anchor reproduction failed for pair {pair_index}")


@torch.inference_mode()
def _render_conversions(
    destination: Path,
    pairs: list[dict],
    cpu_batch: dict[str, Tensor],
    states: dict[str, dict[str, Tensor]],
    model: HARPCore,
    system: HARPFlow,
    vocoder,
    encoder: WavLMSpeakerEncoder,
    references: dict[int, dict],
    config: HARPConfig,
    device: torch.device,
    steps: int,
    method: str,
) -> list[dict]:
    import soundfile as sf

    batch = {name: value.to(device) for name, value in cpu_batch.items()}
    results = []
    for state_name in ("raw", "ema"):
        model.load_state_dict(states[state_name], strict=True)
        model.eval()
        prediction = system.sample(
            batch["content"],
            batch["f0"],
            batch["rms"],
            batch["speaker"],
            batch["mask"],
            steps=steps,
            method=method,
            guidance_strength=1.0,
            initial_noise=_fixed_noise(batch, config.model.mel_channels, device),
        )
        waveforms = []
        for pair_index, _pair in enumerate(pairs):
            frames = int(batch["length"][pair_index])
            waveform = synthesize_pc_nsf(
                vocoder,
                prediction[pair_index, :frames],
                batch["f0"][pair_index, :frames, 0],
                device,
            )
            name = f"pair-{pair_index:02d}-{state_name}.wav"
            sf.write(
                destination / name,
                waveform.numpy(),
                config.feature.sample_rate,
                subtype="PCM_24",
            )
            waveforms.append(waveform)
        embeddings = encoder.encode(waveforms, config.feature.sample_rate)
        paired = zip(pairs, embeddings, strict=True)
        for pair_index, (pair, embedding) in enumerate(paired):
            reference = references[pair_index]
            metrics = speaker_metrics(
                embedding,
                reference["source_prototype"],
                reference["target_prototype"],
            )
            positive = reference["anchor_separation"] > 0
            progress = (
                normalized_progress(
                    metrics["target_margin"],
                    reference["source_margin"],
                    reference["target_margin"],
                )
                if positive
                else None
            )
            results.append(
                {
                    "state": state_name,
                    "pair": pair_index,
                    "source_speaker": pair["source_speaker"],
                    "target_speaker": pair["target_speaker"],
                    "generated": f"pair-{pair_index:02d}-{state_name}.wav",
                    "positive_anchor": positive,
                    "anchor_separation": reference["anchor_separation"],
                    **metrics,
                    "normalized_progress": progress,
                }
            )
    return results


def _aggregate(results: list[dict]) -> dict[str, dict[str, float | int]]:
    output = {}
    for state in ("raw", "ema"):
        selected = [
            item
            for item in results
            if item["state"] == state and item["positive_anchor"]
        ]
        progress = torch.tensor([item["normalized_progress"] for item in selected])
        output[state] = {
            "positive_anchor_pairs": len(selected),
            "normalized_progress_mean": float(progress.mean()),
            "normalized_progress_median": float(progress.median()),
            "normalized_progress_p10": float(torch.quantile(progress, 0.1)),
            "normalized_progress_p90": float(torch.quantile(progress, 0.9)),
            "target_margin_mean": sum(item["target_margin"] for item in selected)
            / len(selected),
        }
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
