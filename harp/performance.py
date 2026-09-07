import torch
from torch import nn


def resolve_device(spec: str) -> torch.device:
    """Resolve the CLI device spec while keeping explicit requests strict."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(spec)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            f"invalid device {spec!r}; use auto, cpu, or a torch device"
        ) from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{spec} requested but CUDA is unavailable; use --device auto or cpu"
        )
    return device


def configure_cuda(
    device: torch.device, *, sdpa_backend: str, allow_tf32: bool
) -> dict[str, object]:
    if device.type != "cuda":
        return {
            "device": str(device),
            "precision": "fp32",
            "sdpa_backend": "eager",
            "cudnn_sdpa": False,
        }
    if sdpa_backend != "cudnn":
        raise ValueError("unsupported SDPA backend")
    capability = torch.cuda.get_device_capability(device)
    ampere_or_newer = capability >= (8, 0)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if ampere_or_newer:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(True)
        attention_backend = "cudnn"
    else:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        attention_backend = "auto"
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "compute_capability": capability,
        "precision": "bf16" if ampere_or_newer else "fp16",
        "sdpa_backend": attention_backend,
        "cudnn_sdpa": ampere_or_newer,
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
