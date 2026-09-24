"""Convert Pydantic JSON schemas into the subset accepted by OpenAI *strict* structured outputs.

Strict mode rejects a schema unless every object lists **all** of its properties in
`required` and sets `additionalProperties: false`. Pydantic omits fields with a
default from `required`, so a model such as `candidates: list[...] = []` is
rejected with "'required' is required to be supplied and to be an array
including every key in properties". This module rewrites the schema in place of
the caller's copy so the response models can keep their Python defaults.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

# Keywords strict mode does not accept on any node.
_UNSUPPORTED_KEYWORDS = frozenset({"default", "minLength", "maxLength", "examples"})


def _make_nullable(node: dict[str, Any]) -> None:
    """Allow `null` for a property that used to be optional, so the model can omit a value."""
    if "type" in node:
        types = node["type"] if isinstance(node["type"], list) else [node["type"]]
        if "null" not in types:
            node["type"] = [*types, "null"]
        return
    if "anyOf" in node:
        if not any(alt.get("type") == "null" for alt in node["anyOf"]):
            node["anyOf"].append({"type": "null"})
        return
    if "$ref" in node:
        node["anyOf"] = [{"$ref": node.pop("$ref")}, {"type": "null"}]


def _walk(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _walk(item)
        return
    if not isinstance(node, dict):
        return

    if node.get("type") == "object" or "properties" in node:
        properties: dict[str, Any] = node.setdefault("properties", {})
        required = set(node.get("required", []))
        for name, prop in properties.items():
            if name not in required and isinstance(prop, dict) and prop.get("default", ...) is None:
                # `x: T | None = None` -> keep nullable so "no value" is still expressible.
                _make_nullable(prop)
        node["required"] = list(properties)
        node["additionalProperties"] = False

    for key in _UNSUPPORTED_KEYWORDS:
        node.pop(key, None)

    for key, value in node.items():
        if key in {"properties", "$defs", "definitions"} and isinstance(value, dict):
            for child in value.values():
                _walk(child)
        elif key in {"items", "anyOf", "oneOf", "allOf", "prefixItems"}:
            _walk(value)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return `model`'s JSON schema rewritten for OpenAI strict structured outputs."""
    schema = copy.deepcopy(model.model_json_schema())
    _walk(schema)
    return schema


__all__ = ["strict_json_schema"]
