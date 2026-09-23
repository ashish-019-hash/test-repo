"""Write stage outputs as canonical JSON files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from doc_extractor.storage.canonical_json import dumps


def write_json(path: Path, payload: Any, float_precision: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(payload, float_precision), encoding="utf-8")
    return path


def write_models(
    out_dir: Path, name: str, models: Sequence[BaseModel] | BaseModel, float_precision: int = 4
) -> Path:
    """Write `<out_dir>/<name>.json`. Lists are serialized as JSON arrays."""
    payload: Any = models if isinstance(models, BaseModel) else list(models)
    return write_json(Path(out_dir) / f"{name}.json", payload, float_precision)
