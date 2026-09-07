import pytest
import torch

from harp.performance import resolve_device


def test_auto_device_uses_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_auto_device_uses_cuda_when_cuda_is_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


def test_explicit_cpu_device_is_preserved() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_explicit_cuda_device_fails_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device("cuda")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid device"):
        resolve_device("not-a-device")
