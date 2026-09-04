from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    dataset: str
    speaker: str
    song: str
    feature_prefix: str
    frames: int
    split: str
    quality_status: str
    content_feature_path: str | None = None

    @property
    def speaker_key(self) -> str:
        return f"{self.dataset}:{self.speaker}"

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ManifestEntry:
        required = {
            "id",
            "dataset",
            "speaker",
            "song",
            "feature_prefix",
            "frames",
            "split",
            "quality_status",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"manifest entry omits {sorted(missing)}")
        return cls(
            id=str(payload["id"]),
            dataset=str(payload["dataset"]),
            speaker=str(payload["speaker"]),
            song=str(payload["song"]),
            feature_prefix=str(payload["feature_prefix"]),
            frames=int(payload["frames"]),
            split=str(payload["split"]),
            quality_status=str(payload["quality_status"]),
            content_feature_path=(
                None
                if payload.get("content_feature_path") is None
                else str(payload["content_feature_path"])
            ),
        )


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    entries = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = ManifestEntry.from_dict(json.loads(line))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid manifest line {line_number}: {error}"
                ) from error
            if entry.id in seen:
                raise ValueError(f"duplicate manifest id: {entry.id}")
            if entry.frames <= 0:
                raise ValueError(f"{entry.id}: frames must be positive")
            seen.add(entry.id)
            entries.append(entry)
    if not entries:
        raise ValueError("manifest is empty")
    return entries


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
