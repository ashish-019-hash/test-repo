"""Shared pipeline state (LangGraph) and bookkeeping records."""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import Field

from doc_extractor.schemas.attribute import Attribute
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.common import StrictModel
from doc_extractor.schemas.document import Document
from doc_extractor.schemas.entity import CanonicalEntity, Entity, NormalizedEntity
from doc_extractor.schemas.mapping import Mapping
from doc_extractor.schemas.output import FinalOutput
from doc_extractor.schemas.review import DuplicateGroup, ReviewDecision


class StageName(StrEnum):
    document_processing = "document_processing"
    chunking = "chunking"
    attribute_extraction = "attribute_extraction"
    attribute_storage = "attribute_storage"
    entity_generation = "entity_generation"
    entity_normalization = "entity_normalization"
    entity_reviewer = "entity_reviewer"
    attribute_mapping = "attribute_mapping"
    final_output = "final_output"


STAGE_ORDER: tuple[StageName, ...] = tuple(StageName)

StageStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


class StageRecord(StrictModel):
    stage: StageName
    status: StageStatus
    attempts: int = 0
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: int | None = None


class TraceRecord(StrictModel):
    stage: StageName
    event: str
    data: dict[str, Any] = Field(default_factory=dict)
    llm_calls: int = 0
    cache_hits: int = 0
    elapsed_ms: int = 0


ReviewItemKind = Literal["attribute_review", "possible_duplicate", "entity_review"]


class HumanReviewItem(StrictModel):
    item_id: str = Field(description='"hri-<sha256(kind|sorted ref_ids)[:16]>"')
    kind: ReviewItemKind
    ref_ids: list[str]
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict)


def merge_dict(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    return {**(a or {}), **(b or {})}


class PipelineState(TypedDict, total=False):
    # inputs
    input_path: str
    out_dir: str
    config_hash: str
    resume_from: str | None
    # stage outputs (each set exactly once by its owning agent)
    document: Document
    chunks: list[Chunk]
    attributes: list[Attribute]
    discarded_attributes: list[Attribute]
    attributes_path: str
    entities: list[Entity]
    normalized_entities: list[NormalizedEntity]
    duplicate_groups: list[DuplicateGroup]
    canonical_entities: list[CanonicalEntity]
    review_decisions: list[ReviewDecision]
    mappings: list[Mapping]
    unmapped_attribute_ids: list[str]
    final_output: FinalOutput
    # bookkeeping (reducers)
    stages: Annotated[dict[str, StageRecord], merge_dict]
    traces: Annotated[list[TraceRecord], operator.add]
    human_review_queue: Annotated[list[HumanReviewItem], operator.add]
