from __future__ import annotations

import torch
from torch import nn


def configure_cuda(
    device: torch.device, *, sdpa_backend: str, allow_tf32: bool
) -> dict[str, object]:
    if device.type != "cuda":
        return {"device": str(device), "cudnn_sdpa": False}
    if sdpa_backend != "cudnn":
        raise ValueError("HARP v1 supports only the configured cuDNN SDPA backend")
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    torch.backends.cudnn.allow_tf32 = allow_tf32
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
        "allow_tf32": allow_tf32,
    }


def compile_model_in_place(
    model: nn.Module,
    mode: str,
    *,
    epilogue_fusion: bool = True,
    shape_padding: bool = True,
) -> None:
    torch._inductor.config.epilogue_fusion = epilogue_fusion
    torch._inductor.config.shape_padding = shape_padding
    keys = tuple(model.state_dict())
    model.compile(mode=mode, dynamic=False, fullgraph=True)
    if tuple(model.state_dict()) != keys:
        raise RuntimeError("torch.compile changed model state keys")
