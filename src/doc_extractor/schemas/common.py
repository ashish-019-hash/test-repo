"""Shared value objects used across all stage schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base for every schema: immutable, no unknown fields, alias-aware."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class Provenance(StrictModel):
    """Where a record came from inside the source file."""

    file: str = Field(description="Base file name only (never an absolute path).")
    page: int
    block_id: str
    bbox: tuple[float, float, float, float] | None = None


class Evidence(StrictModel):
    """A verbatim quote from a chunk. `text` MUST be a substring of the chunk's source_text."""

    chunk_id: str
    text: str
    start: int | None = None
    end: int | None = None


class Unit(StrictModel):
    raw: str | None = None
    base: str | None = None
    factor: float | None = None


class ValueDomain(StrictModel):
    kind: Literal["range", "enum", "pattern", "boolean", "free"]
    from_: float | str | None = Field(default=None, alias="from")
    to: float | str | None = None
    values: list[str] = Field(default_factory=list)
    pattern: str | None = None


Layer = Literal["CFS", "RFS", "Resource", "Party", "Other"]
