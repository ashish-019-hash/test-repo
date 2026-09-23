"""Schemas for Entity Generation, Normalization, and canonical entities produced by the Reviewer."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from doc_extractor.schemas.common import Evidence, Layer, StrictModel

EntityOrigin = Literal["binding", "promotion", "lexicon", "llm"]


class Entity(StrictModel):
    entity_id: str = Field(description='"ent-<sha256(doc|entity_name|entity_type)[:16]>"')
    entity_name: str
    entity_type: str
    layer: Layer = "Other"
    source_attributes: list[str] = Field(default_factory=list)
    source_chunks: list[str] = Field(default_factory=list)
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    origin: EntityOrigin
    confidence: float = Field(ge=0.0, le=1.0)


class NormalizedEntity(StrictModel):
    entity_id: str
    original_entity_name: str
    normalized_entity_name: str
    steps: list[str] = Field(default_factory=list, description="Ordered list of applied normalization steps.")
    reasoning: str = ""


class CanonicalEntity(StrictModel):
    canonical_id: str = Field(description='"cent-<sha256(sorted member_ids joined by |)[:16]>"')
    canonical_name: str
    normalized_name: str
    entity_type: str
    layer: Layer = "Other"
    member_ids: list[str] = Field(min_length=1)
    source_attributes: list[str] = Field(default_factory=list)
    source_chunks: list[str] = Field(default_factory=list)
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
