"""Schemas for the Entity Reviewer Agent (duplicate detection and quality validation)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from doc_extractor.schemas.common import StrictModel

DuplicateClass = Literal["DUPLICATE", "POSSIBLE_DUPLICATE", "DISTINCT"]
ValidationStatus = Literal["ACCEPT", "REVIEW", "REJECT"]

QUALITY_CRITERIA: tuple[str, ...] = (
    "specificity",
    "uniqueness",
    "real_world_correspondence",
    "contextual_relevance",
    "completeness",
    "source_grounding",
    "entity_type_correctness",
    "disambiguation_potential",
)


class DuplicatePair(StrictModel):
    a: str = Field(description="entity_id (lexicographically smaller)")
    b: str = Field(description="entity_id (lexicographically larger)")
    features: dict[str, float] = Field(default_factory=dict)
    composite: float = Field(ge=0.0, le=1.0)
    classification: DuplicateClass
    reason: str


class DuplicateGroup(StrictModel):
    canonical_id: str
    canonical_name: str
    member_ids: list[str] = Field(min_length=1)
    pairs: list[DuplicatePair] = Field(
        default_factory=list, description="All non-DISTINCT pairs among members."
    )


class CriterionScore(StrictModel):
    criterion: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class ReviewDecision(StrictModel):
    entity_id: str = Field(description="canonical_id of the reviewed entity")
    criterion_scores: list[CriterionScore]
    overall_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    validation_status: ValidationStatus
    issues: list[str] = Field(default_factory=list)
    reason: str
