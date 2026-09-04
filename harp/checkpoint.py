from __future__ import annotations

from typing import Any

from .config import HARPConfig
from .optimizer import parameter_role_manifest


def checkpoint_contract(
    config: HARPConfig, model: Any, transform_sha256: str
) -> dict[str, Any]:
    config.validate()
    return {
        "model_family": "rift-harp",
        "checkpoint_schema": 1,
        "architecture_version": 1,
        "flow_contract_version": 1,
        "feature_contract_version": 1,
        "config_sha256": config.digest(),
        "flow_transform_sha256": transform_sha256,
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
    }


def validate_checkpoint_contract(payload: dict[str, Any], config: HARPConfig) -> None:
    if payload.get("model_family") != "rift-harp":
        raise ValueError("checkpoint is not a RIFT-HARP checkpoint")
    if payload.get("checkpoint_schema") != 1:
        raise ValueError("unsupported RIFT-HARP checkpoint schema")
    if payload.get("config_sha256") != config.digest():
        raise ValueError("checkpoint config hash does not match this run")
