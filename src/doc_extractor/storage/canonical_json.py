"""Canonical JSON serialization: sorted keys, sorted id-lists, rounded floats, trailing newline.

Two runs over the same input must produce byte-identical files, so every deterministic
output goes through `dumps` here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

_ID_KEYS = (
    "attribute_id",
    "canonical_id",
    "entity_id",
    "mapping_id",
    "chunk_id",
    "block_id",
    "item_id",
    "document_id",
    "page_number",
    "criterion",
    "name",
)


def _sort_key_for_dict(d: dict[str, Any]) -> Any:
    for k in _ID_KEYS:
        if k in d:
            v = d[k]
            return (0, 0, v, "") if isinstance(v, int) and not isinstance(v, bool) else (0, 1, 0, str(v))
    if "from_id" in d and "to_id" in d:
        return (0, 1, 0, f"{d.get('kind')}|{d['from_id']}|{d['to_id']}")
    if "a" in d and "b" in d:
        return (0, 1, 0, f"{d['a']}|{d['b']}")
    # fall back to canonical text; stable but arbitrary
    return (1, 1, 0, json.dumps(d, sort_keys=True, ensure_ascii=False, default=str))


def normalize(obj: Any, float_precision: int = 4) -> Any:
    """Recursively round floats and sort lists of dicts by their id key."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json", by_alias=True)
    if isinstance(obj, bool) or obj is None or isinstance(obj, int | str):
        return obj
    if isinstance(obj, float):
        return round(obj, float_precision)
    if isinstance(obj, dict):
        return {str(k): normalize(v, float_precision) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        items = [normalize(v, float_precision) for v in obj]
        if items and all(isinstance(i, dict) for i in items):
            # Do not reorder positional data (table rows, heading paths); only reorder record lists.
            if _is_record_list(items):
                items.sort(key=_sort_key_for_dict)
        elif isinstance(obj, set | frozenset):
            items.sort(key=str)
        return items
    return str(obj)


def _is_record_list(items: Sequence[dict[str, Any]]) -> bool:
    return any(any(k in d for k in _ID_KEYS) or ("a" in d and "b" in d) or ("from_id" in d) for d in items)


def dumps(obj: Any, float_precision: int = 4) -> str:
    payload = normalize(obj, float_precision)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, separators=(",", ": ")) + "\n"


def loads(text: str) -> Any:
    return json.loads(text)
