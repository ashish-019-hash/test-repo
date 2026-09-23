"""Agent 6: EntityNormalizationAgent.

Pure per-entity normalization. No merging, no dropping: output length always equals
input length and every entity_id is preserved.
"""

from __future__ import annotations

from typing import Any, ClassVar

from doc_extractor.agents.base import BaseAgent
from doc_extractor.exceptions import StageValidationError
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.reviewer import load_lexicons
from doc_extractor.reviewer.normalization import normalize_entity_name
from doc_extractor.schemas.entity import Entity, NormalizedEntity
from doc_extractor.schemas.state import PipelineState, StageName


class EntityNormalizationAgent(BaseAgent):
    name: ClassVar[StageName] = StageName.entity_normalization
    requires: ClassVar[tuple[str, ...]] = ("entities",)
    produces: ClassVar[tuple[str, ...]] = ("normalized_entities",)

    def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
        entities: list[Entity] = state["entities"]
        lexicons = load_lexicons(self.cfg)
        normalized: list[NormalizedEntity] = []
        for entity in entities:
            normalized_name, steps = normalize_entity_name(entity.entity_name, lexicons.aliases)
            reasoning = "; ".join(steps) if steps else "no change"
            normalized.append(
                NormalizedEntity(
                    entity_id=entity.entity_id,
                    original_entity_name=entity.entity_name,
                    normalized_entity_name=normalized_name,
                    steps=steps,
                    reasoning=reasoning,
                )
            )
        trace.count("entities_normalized", len(normalized))
        return {"normalized_entities": normalized}

    def validate_output(self, delta: dict[str, Any], state: PipelineState) -> None:
        super().validate_output(delta, state)
        normalized: list[NormalizedEntity] = delta["normalized_entities"]
        entities: list[Entity] = state["entities"]
        if len(normalized) != len(entities):
            raise StageValidationError(
                str(self.name),
                f"normalized_entities length {len(normalized)} != entities length {len(entities)}",
            )
        entity_ids = {e.entity_id for e in entities}
        normalized_ids = {n.entity_id for n in normalized}
        if entity_ids != normalized_ids:
            raise StageValidationError(str(self.name), "normalized_entities entity_ids do not match entities")
