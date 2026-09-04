from __future__ import annotations

import torch
from torch import nn


def configure_cuda(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"device": str(device), "cudnn_sdpa": False}
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(True)
    if not torch.backends.cuda.cudnn_sdp_enabled():
        raise RuntimeError("cuDNN SDPA could not be enabled")
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_sdpa": True,
    }


def compile_model_in_place(model: nn.Module, mode: str) -> None:
    keys = tuple(model.state_dict())
    model.compile(mode=mode, dynamic=False)
    if tuple(model.state_dict()) != keys:
        raise RuntimeError("torch.compile changed model state keys")
