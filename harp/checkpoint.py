import hashlib
from pathlib import Path
from typing import Any

from .config import HARPConfig
from .optimizer import parameter_role_manifest


def checkpoint_contract(
    config: HARPConfig,
    model: Any,
    *,
    transform_sha256: str,
    transform_audit_sha256: str,
    feature_contract_sha256: str,
    manifest_sha256: str,
    exposure_semantics_sha256: str,
    batch_runtime_sha256: str,
    sampling_audit_sha256: str,
    runtime: dict[str, Any],
    source_control: dict[str, Any],
) -> dict[str, Any]:
    config.validate()
    return {
        "model_family": "rift-harp",
        "checkpoint_schema": 1,
        "architecture_version": config.contract.architecture_version,
        "flow_contract_version": config.contract.flow_contract_version,
        "feature_contract_version": config.contract.feature_contract_version,
        "config_sha256": config.digest(),
        "flow_transform_sha256": transform_sha256,
        "flow_transform_audit_sha256": transform_audit_sha256,
        "feature_contract_sha256": feature_contract_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "exposure_semantics_sha256": exposure_semantics_sha256,
        "batch_runtime_sha256": batch_runtime_sha256,
        "sampling_audit_sha256": sampling_audit_sha256,
        "optimizer_role_map_version": config.contract.optimizer_role_map_version,
        "optimizer_roles": parameter_role_manifest(model),
        "numeric_contract": {
            "flow_coefficient_dtype": "float32",
            "ode_state_dtype": "float32",
            "model_compute": "bfloat16",
            "cfg_domain": "residual",
            "lambda_floor": config.flow.lambda_floor,
            "q_floor": config.flow.q_floor,
        },
        "runtime": runtime,
        "source_control": source_control,
    }


def validate_checkpoint_contract(payload: dict[str, Any], config: HARPConfig) -> None:
    if payload.get("model_family") != "rift-harp":
        raise ValueError("checkpoint is not a RIFT-HARP checkpoint")
    if payload.get("checkpoint_schema") != 1:
        raise ValueError("unsupported RIFT-HARP checkpoint schema")
    if payload.get("config_sha256") != config.digest():
        raise ValueError("checkpoint config hash does not match this run")


def validate_contract_identity(
    payload: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Validate immutable training meaning while allowing runtime drift."""
    keys = (
        "model_family",
        "checkpoint_schema",
        "architecture_version",
        "flow_contract_version",
        "feature_contract_version",
        "config_sha256",
        "flow_transform_sha256",
        "flow_transform_audit_sha256",
        "feature_contract_sha256",
        "dataset_manifest_sha256",
        "exposure_semantics_sha256",
        "batch_runtime_sha256",
        "sampling_audit_sha256",
        "optimizer_role_map_version",
        "optimizer_roles",
        "numeric_contract",
        "source_control",
    )
    mismatched = [key for key in keys if payload.get(key) != expected.get(key)]
    if mismatched:
        raise ValueError(f"checkpoint/run contract differs in {mismatched}")


def validate_external_artifacts(
    payload: dict[str, Any],
    *,
    transform_path: str | Path,
    transform_audit_path: str | Path,
    feature_contract_path: str | Path,
    manifest_path: str | Path,
) -> None:
    paths = {
        "flow_transform_sha256": Path(transform_path),
        "flow_transform_audit_sha256": Path(transform_audit_path),
        "feature_contract_sha256": Path(feature_contract_path),
        "dataset_manifest_sha256": Path(manifest_path),
    }
    mismatched = []
    for key, path in paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if payload.get(key) != digest:
            mismatched.append(key)
    if mismatched:
        raise ValueError(f"checkpoint external artifacts differ in {mismatched}")
