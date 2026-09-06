from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .model import HARPCore


_HEAVY_LINEAR_SUFFIXES = (
    ".attention.qkv",
    ".attention.output",
    ".feed_forward.input",
    ".feed_forward.output",
)


def heavy_linear_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.startswith("blocks.")
        and name.endswith(_HEAVY_LINEAR_SUFFIXES)
    )


def configure_heavy_linears(model: HARPCore, precision: str) -> dict[str, object]:
    if precision != "float8_rowwise":
        raise ValueError(f"unsupported heavy Linear precision: {precision}")
    expected_names = heavy_linear_names(model)
    expected_count = model.config.depth * len(_HEAVY_LINEAR_SUFFIXES)
    if len(expected_names) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} heavy Linear modules, found {expected_names}"
        )
    parameter_device = next(model.parameters()).device
    if parameter_device.type != "cuda":
        return {
            "heavy_linear_precision": precision,
            "heavy_linear_runtime": "high_precision_cpu_fallback",
            "heavy_linear_count": 0,
            "heavy_linear_names": expected_names,
            "torchao": None,
        }
    state_keys = tuple(model.state_dict())
    try:
        import torchao
        from torchao.float8 import (
            Float8LinearConfig,
            Float8LinearRecipeName,
            convert_to_float8_training,
        )
        from torchao.float8.float8_linear import Float8Linear
    except ImportError as error:
        raise RuntimeError(
            "float8_rowwise requires a Torch-compatible torchao installation"
        ) from error

    selected = set(expected_names)
    convert_to_float8_training(
        model,
        module_filter_fn=lambda _module, fqn: fqn in selected,
        config=Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.ROWWISE),
    )
    converted = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, Float8Linear)
    )
    if converted != expected_names:
        raise RuntimeError(
            f"FP8 conversion differs from the frozen heavy Linear set: {converted}"
        )
    if tuple(model.state_dict()) != state_keys:
        raise RuntimeError("FP8 conversion changed model state keys")
    non_fp32 = {
        name: str(parameter.dtype)
        for name, parameter in model.named_parameters()
        if parameter.dtype != torch.float32
    }
    if non_fp32:
        raise RuntimeError(f"FP8 conversion changed master parameter dtype: {non_fp32}")
    return {
        "heavy_linear_precision": precision,
        "heavy_linear_runtime": "torchao_float8_linear",
        "heavy_linear_count": len(converted),
        "heavy_linear_names": converted,
        "torchao": getattr(torchao, "__version__", "unknown"),
    }
