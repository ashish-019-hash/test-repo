"""Schemas for the Final Output Agent and run metadata."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from doc_extractor.schemas.attribute import Attribute
from doc_extractor.schemas.common import StrictModel
from doc_extractor.schemas.document import DocumentSummary
from doc_extractor.schemas.entity import CanonicalEntity
from doc_extractor.schemas.mapping import Mapping
from doc_extractor.schemas.review import DuplicateGroup, ReviewDecision

TraceKind = Literal[
    "document>chunk",
    "chunk>attribute",
    "attribute>entity",
    "entity>canonical",
    "attribute>mapping",
    "mapping>entity",
]


class TraceEdge(StrictModel):
    from_id: str
    to_id: str
    kind: TraceKind


class ReviewSection(StrictModel):
    duplicates_removed: list[DuplicateGroup] = Field(default_factory=list)
    entities_requiring_review: list[ReviewDecision] = Field(default_factory=list)
    rejected_entities: list[ReviewDecision] = Field(default_factory=list)


class FinalOutput(StrictModel):
    schema_version: str = "1.0"
    document: DocumentSummary
    attributes: list[Attribute] = Field(default_factory=list)
    entities: list[CanonicalEntity] = Field(default_factory=list)
    mappings: list[Mapping] = Field(default_factory=list)
    review: ReviewSection = Field(default_factory=ReviewSection)
    unmapped_attribute_ids: list[str] = Field(default_factory=list)
    dangling_attribute_ids: list[str] = Field(default_factory=list)
    trace: list[TraceEdge] = Field(default_factory=list)


class RunMetadata(StrictModel):
    """Written to run.json only. The only deterministic-output-adjacent file allowed to hold timestamps."""

    run_id: str
    document_id: str | None = None
    input_file: str
    started_at: str
    finished_at: str | None = None
    provider: str
    model: str | None = None
    system_fingerprint: str | None = None
    config_hash: str
    package_version: str
    status: Literal["succeeded", "failed"]
    thread_id: str | None = None
    stages: dict[str, Any] = Field(default_factory=dict)
    traces: list[dict[str, Any]] = Field(default_factory=list)
