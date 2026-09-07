import hashlib
import os
import tempfile
from pathlib import Path

import torch


def load_compiler_cache_artifact(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"compiler cache artifact does not exist: {path}")
    payload = path.read_bytes()
    info = torch.compiler.load_cache_artifacts(payload)
    return {
        "compiler_cache_artifact": str(path.resolve()),
        "compiler_cache_artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "compiler_cache_load_info": None if info is None else str(info),
    }


def save_compiler_cache_artifact(path: Path) -> dict[str, object]:
    result = torch.compiler.save_cache_artifacts()
    if result is None:
        raise RuntimeError("torch.compiler produced no cache artifacts")
    payload, info = result
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return {
        "compiler_cache_artifact": str(path.resolve()),
        "compiler_cache_artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "compiler_cache_save_info": str(info),
        "compiler_cache_artifact_bytes": len(payload),
    }
