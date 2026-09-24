"""Tests for `doc_extractor.llm.schema.strict_json_schema`."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from doc_extractor.llm import tasks
from doc_extractor.llm.schema import strict_json_schema


def _objects(node: Any) -> list[dict[str, Any]]:
    """Every object-typed schema node, including nested `$defs`, items and anyOf members."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            found.append(node)
        for value in node.values():
            found.extend(_objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_objects(item))
    return found


def test_every_task_schema_is_strict() -> None:
    """Azure strict mode: all properties required and no additional properties, at every level."""
    for task in tasks.TASKS.values():
        schema = strict_json_schema(task.response_model)
        objects = _objects(schema)
        assert objects, task.name
        for obj in objects:
            assert obj["additionalProperties"] is False, task.name
            assert sorted(obj["required"]) == sorted(obj["properties"]), task.name
        assert "default" not in str(schema), task.name


def test_optional_fields_become_required_but_nullable() -> None:
    class Inner(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str = Field(default="x", min_length=1, max_length=5)

    class Outer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        note: str | None = None
        inner: Inner | None = None
        items: list[Inner] = []

    schema = strict_json_schema(Outer)

    assert schema["required"] == ["note", "inner", "items"]
    assert schema["additionalProperties"] is False
    # `str | None = None` keeps its null alternative.
    assert {"type": "null"} in schema["properties"]["note"]["anyOf"]
    # A `$ref | None = None` keeps its null alternative too.
    assert {"type": "null"} in schema["properties"]["inner"]["anyOf"]
    # A non-null default just becomes required; the (unsupported) length keywords are dropped.
    inner = schema["$defs"]["Inner"]
    assert inner["required"] == ["value"]
    assert "default" not in inner["properties"]["value"]
    assert "minLength" not in inner["properties"]["value"]
    assert "maxLength" not in inner["properties"]["value"]


def test_original_pydantic_schema_is_not_mutated() -> None:
    before = tasks.AttributeCandidatesResponse.model_json_schema()
    strict_json_schema(tasks.AttributeCandidatesResponse)
    assert tasks.AttributeCandidatesResponse.model_json_schema() == before
    # Pydantic itself still omits the defaulted field from `required`.
    assert "required" not in before
