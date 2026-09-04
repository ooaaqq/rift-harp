from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch import Tensor, nn

from .model import HARPCore


@torch.no_grad()
def optimizer_role_telemetry(
    model: HARPCore,
    optimizer: torch.optim.Optimizer,
    *,
    clip_coefficient: float,
) -> dict[str, dict[str, float]]:
    roles = {id(parameter): role for _, parameter, role in model.parameter_roles()}
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for group in optimizer.param_groups:
        learning_rate = float(group["lr"])
        eps = float(group["eps"])
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            role = roles[id(parameter)]
            count = parameter.numel()
            totals[role]["parameter_count"] += count
            totals[role]["parameter_square_sum"] += float(
                parameter.float().square().sum()
            )
            if parameter.grad is not None:
                gradient = parameter.grad.float()
                totals[role]["gradient_count"] += count
                totals[role]["gradient_square_sum"] += float(gradient.square().sum())
            state = optimizer.state.get(parameter, {})
            if "exp_avg" not in state:
                continue
            moment = state["exp_avg"].float()
            second = state["exp_avg_sq"].float()
            step = float(state["step"])
            totals[role]["moment_count"] += count
            totals[role]["moment_abs_sum"] += float(moment.abs().sum())
            totals[role]["second_sum"] += float(second.sum())
            corrected_moment = moment / (1 - beta1**step)
            corrected_second = second / (1 - beta2**step)
            update = learning_rate * corrected_moment / (corrected_second.sqrt() + eps)
            totals[role]["update_square_sum"] += float(update.square().sum())

    result = {}
    total_pre_clip_gradient_square = (
        sum(values["gradient_square_sum"] for values in totals.values())
        / max(clip_coefficient, 1e-12) ** 2
    )
    for role, values in sorted(totals.items()):
        parameter_count = max(1.0, values["parameter_count"])
        gradient_count = max(1.0, values["gradient_count"])
        moment_count = max(1.0, values["moment_count"])
        parameter_rms = math.sqrt(values["parameter_square_sum"] / parameter_count)
        clipped_gradient_rms = math.sqrt(values["gradient_square_sum"] / gradient_count)
        update_rms = math.sqrt(values["update_square_sum"] / moment_count)
        result[role] = {
            "parameter_rms": parameter_rms,
            "pre_clip_gradient_rms": clipped_gradient_rms
            / max(clip_coefficient, 1e-12),
            "pre_clip_gradient_norm": math.sqrt(values["gradient_square_sum"])
            / max(clip_coefficient, 1e-12),
            "pre_clip_gradient_norm_fraction": (
                values["gradient_square_sum"]
                / max(clip_coefficient, 1e-12) ** 2
                / max(total_pre_clip_gradient_square, 1e-24)
            ),
            "adam_m_abs_mean": values["moment_abs_sum"] / moment_count,
            "adam_sqrt_v_rms": math.sqrt(values["second_sum"] / moment_count),
            "preconditioned_update_rms": update_rms,
            "update_to_weight_rms": update_rms / max(parameter_rms, 1e-12),
        }
    return result


def model_activation_telemetry(
    model: HARPCore, model_inputs: tuple[Tensor, ...]
) -> dict[str, float]:
    """Run a small eager probe without creating another compiled graph."""
    mask = model_inputs[-1].bool()
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    handles = []

    def capture(name: str):
        def hook(
            _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor
        ) -> None:
            value = output.detach()
            # TorchAO Float8Tensor subclasses only implement a restricted set of
            # shape operations.  Observability belongs at the semantic output
            # boundary, so leave the wrapper before masking or reshaping.
            if type(value).__module__.startswith("torchao."):
                value = value.dequantize()
            value = value.float()
            selection = mask
            if value.ndim == 3 and value.shape[1] == mask.shape[1]:
                value = value[selection]
            square_sum = float(value.square().sum())
            totals[name][0] += square_sum
            totals[name][1] += value.numel()

        return hook

    named_modules = {
        "state_projection": model.state_input,
        "content": model.content_input,
        "pitch": model.pitch_input,
        "harmonic": model.harmonic_input,
        "energy": model.energy_input,
        "frame_condition": model.frame_condition,
        "input_residual": model.input_mix,
    }
    for name, module in named_modules.items():
        handles.append(module.register_forward_hook(capture(name)))
    for index, block in enumerate(model.blocks, start=1):
        handles.append(
            block.modulation.register_forward_hook(capture("adaln_modulation"))
        )
        handles.append(block.attention.register_forward_hook(capture("attention")))
        handles.append(block.feed_forward.register_forward_hook(capture("ff")))
        handles.append(block.register_forward_hook(capture("residual_stream")))
        if str(index) in model.harmonic_adapters:
            adapter = model.harmonic_adapters[str(index)]
            handles.append(
                adapter.register_forward_hook(capture(f"harmonic_adapter_{index}"))
            )
    try:
        device_type = next(model.parameters()).device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            model.forward(*model_inputs)
    finally:
        for handle in handles:
            handle.remove()
    result = {
        name: math.sqrt(square_sum / max(count, 1.0))
        for name, (square_sum, count) in sorted(totals.items())
    }
    stream_rms = result.get("residual_stream", 0.0)
    for index in model.config.harmonic_injection_blocks:
        key = f"harmonic_adapter_{index}"
        result[f"{key}_to_residual_stream"] = result.get(key, 0.0) / max(
            stream_rms, 1e-12
        )
    return result
