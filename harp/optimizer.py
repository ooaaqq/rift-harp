from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn

from .model import HARPCore


@dataclass(frozen=True)
class RoleHyperparameters:
    learning_rate: float
    weight_decay: float


ROLE_HYPERPARAMETERS = {
    "backbone": RoleHyperparameters(1.5e-4, 0.010),
    "ff_expansion": RoleHyperparameters(3.0e-4, 0.005),
    "ff_contraction": RoleHyperparameters(7.5e-5, 0.020),
    "stem": RoleHyperparameters(1.5e-4, 0.010),
    "adaln": RoleHyperparameters(1.5e-4, 0.010),
    "harmonic_encoder": RoleHyperparameters(1.5e-4, 0.010),
    "harmonic_adapter": RoleHyperparameters(1.5e-4, 0.010),
    "output": RoleHyperparameters(1.5e-4, 0.010),
    "speaker": RoleHyperparameters(2.0e-4, 0.0),
    "branch_gain": RoleHyperparameters(1.5e-4, 0.0),
}


def build_optimizer(model: HARPCore, *, fused: bool = True) -> torch.optim.AdamW:
    grouped: dict[tuple[str, bool], list[nn.Parameter]] = defaultdict(list)
    assignments: dict[int, str] = {}
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    for name, parameter, role in model.parameter_roles():
        if not parameter.requires_grad:
            continue
        if role not in ROLE_HYPERPARAMETERS:
            raise ValueError(f"unknown optimizer role {role!r} for {name}")
        identity = id(parameter)
        if identity in assignments:
            raise ValueError(f"parameter {name} assigned more than once")
        assignments[identity] = role
        no_decay = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or role
            in {
                "speaker",
                "branch_gain",
            }
        )
        grouped[(role, no_decay)].append(parameter)
    missing = trainable - assignments.keys()
    extra = assignments.keys() - trainable
    if missing or extra:
        detail = f"missing={len(missing)}, extra={len(extra)}"
        raise ValueError(f"optimizer role coverage failed: {detail}")
    groups = []
    for (role, no_decay), parameters in sorted(grouped.items()):
        hyperparameters = ROLE_HYPERPARAMETERS[role]
        groups.append(
            {
                "params": parameters,
                "lr": hyperparameters.learning_rate,
                "weight_decay": 0.0 if no_decay else hyperparameters.weight_decay,
                "role": role,
                "decay": not no_decay,
            }
        )
    return torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused,
    )


def parameter_role_manifest(model: HARPCore) -> dict[str, str]:
    return {name: role for name, _, role in model.parameter_roles()}
