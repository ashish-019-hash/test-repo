"""Schemas for the Attribute Extraction Agent (candidates, scoring, binding, final attribute)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from doc_extractor.schemas.common import Evidence, Layer, Provenance, StrictModel, Unit, ValueDomain

CandidateOrigin = Literal["spec_table", "kv_pair", "form_field", "prose_regex", "llm"]
Route = Literal["accept", "review", "discard"]
BindingScope = Literal[
    "table_subject", "nearest_heading", "syntactic", "compound_modifier", "document_default", "dangling"
]
AttributeType = Literal[
    "integer", "number", "string", "boolean", "enum", "date", "identifier", "measure", "unknown"
]
Optionality = Literal["M", "O", "C"]


class SignalHit(StrictModel):
    name: str = Field(description='Dotted signal key, e.g. "structural.spec_table_row"')
    weight: float
    evidence: str


class ScoreBreakdown(StrictModel):
    hits: list[SignalHit] = Field(default_factory=list)
    raw_sum: float
    clamped: float = Field(ge=0.0, le=1.0)
    short_circuit: bool = False
    route: Route
    discard_reason: str | None = Field(
        default=None,
        description='Set for discards, e.g. "below_threshold" or "promoted" (has_sub_attributes fired).',
    )


class EntityBinding(StrictModel):
    entity_name: str | None = None
    scope: BindingScope
    structural_distance: int = 0
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str | None = None


class AttributeCandidate(StrictModel):
    """Raw candidate before scoring. Produced by deterministic generators or the LLM."""

    raw_name: str
    chunk_id: str
    block_id: str
    source_text: str
    origin: CandidateOrigin
    cells: dict[str, str] = Field(
        default_factory=dict,
        description="Spec-table cells by normalized header: name,type,units,optionality,range,default,description",
    )
    sentence: str | None = Field(
        default=None, description="Sentence containing the candidate, if from prose."
    )


class Attribute(StrictModel):
    attribute_id: str = Field(description='"attr-<sha256(doc|chunk|attribute_name|source_text)[:16]>"')
    attribute_name: str = Field(description="camelCase normalized name")
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    attribute_type: AttributeType = "unknown"
    entity: str | None = None
    binding: EntityBinding
    layer: Layer | None = None
    unit: Unit | None = None
    cardinality: str | None = None
    optionality: Optionality | None = None
    value_domain: ValueDomain | None = None
    default: str | None = None
    description: str | None = None
    source_document: str
    source_chunk: str
    source_text: str
    provenance: Provenance
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    score: ScoreBreakdown
    origin: CandidateOrigin
