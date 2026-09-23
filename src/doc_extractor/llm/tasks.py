"""Structured-output response models and `LLMTask` definitions.

Providers only ever return instances of these models; every field is exactly what
an agent needs to score/verify a proposal deterministically. `extra="forbid"` so a
provider cannot smuggle unvalidated fields past the schema.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from doc_extractor.llm.base import LLMTask


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateProposal(_Model):
    """A candidate attribute name, grounded in a verbatim quote from the source text."""

    raw_name: str
    source_text_quote: str
    sentence: str | None = None


class AttributeCandidatesResponse(_Model):
    candidates: list[CandidateProposal] = []


class EntityProposal(_Model):
    entity_name: str
    entity_type: str
    chunk_id: str
    evidence_quote: str


class EntityProposalsResponse(_Model):
    entities: list[EntityProposal] = []


class EntityQualityResponse(_Model):
    specificity_delta: float
    real_world_delta: float
    reason: str


class BindingJudgementResponse(_Model):
    entity_name: str | None = None
    evidence_quote: str | None = None
    reason: str


ATTRIBUTE_CANDIDATES_TASK = LLMTask(
    name="attribute_candidates",
    prompt_file="attribute_candidates.md",
    response_model=AttributeCandidatesResponse,
    max_output_tokens=2000,
)

ENTITY_PROPOSALS_TASK = LLMTask(
    name="entity_proposals",
    prompt_file="entity_proposals.md",
    response_model=EntityProposalsResponse,
    max_output_tokens=2000,
)

ENTITY_QUALITY_TASK = LLMTask(
    name="entity_quality",
    prompt_file="entity_quality.md",
    response_model=EntityQualityResponse,
    max_output_tokens=500,
)

BINDING_JUDGEMENT_TASK = LLMTask(
    name="binding_judgement",
    prompt_file="binding_judgement.md",
    response_model=BindingJudgementResponse,
    max_output_tokens=500,
)

TASKS: dict[str, LLMTask] = {
    ATTRIBUTE_CANDIDATES_TASK.name: ATTRIBUTE_CANDIDATES_TASK,
    ENTITY_PROPOSALS_TASK.name: ENTITY_PROPOSALS_TASK,
    ENTITY_QUALITY_TASK.name: ENTITY_QUALITY_TASK,
    BINDING_JUDGEMENT_TASK.name: BINDING_JUDGEMENT_TASK,
}

__all__ = [
    "CandidateProposal",
    "AttributeCandidatesResponse",
    "EntityProposal",
    "EntityProposalsResponse",
    "EntityQualityResponse",
    "BindingJudgementResponse",
    "ATTRIBUTE_CANDIDATES_TASK",
    "ENTITY_PROPOSALS_TASK",
    "ENTITY_QUALITY_TASK",
    "BINDING_JUDGEMENT_TASK",
    "TASKS",
]
