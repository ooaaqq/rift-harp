from collections import defaultdict

import torch
from torch import nn

from .config import OptimizerConfig
from .model import HARPCore


def build_optimizer(
    model: HARPCore, config: OptimizerConfig, *, fused: bool = True
) -> torch.optim.AdamW:
    grouped: dict[tuple[str, bool], list[nn.Parameter]] = defaultdict(list)
    assignments: dict[int, str] = {}
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    for name, parameter, role in model.parameter_roles():
        if not parameter.requires_grad:
            continue
        if role not in config.roles:
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
                "branch_mix",
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
        hyperparameters = config.roles[role]
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
        betas=config.betas,
        eps=config.eps,
        fused=fused,
        foreach=False,
    )


def parameter_role_manifest(model: HARPCore) -> dict[str, str]:
    return {name: role for name, _, role in model.parameter_roles()}
