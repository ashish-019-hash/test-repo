"""Schema for the Attribute-to-Entity Mapping Agent."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from doc_extractor.schemas.common import Evidence, StrictModel

RelationshipType = Literal["has_attribute", "constrains", "references"]


class Mapping(StrictModel):
    mapping_id: str = Field(description='"map-<sha256(attribute_id|entity_id|relationship_type)[:16]>"')
    attribute_id: str
    entity_id: str = Field(description="canonical_id of an eligible canonical entity")
    relationship_type: RelationshipType
    source_chunk: str
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
