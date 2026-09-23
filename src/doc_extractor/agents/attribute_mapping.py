"""Agent 8: AttributeMappingAgent.

No LLM. Every attribute with a resolved binding entity is mapped to at most one
canonical entity, only when that canonical entity is eligible (ACCEPT, or REVIEW when
`cfg.mapping.include_review_entities` is set) and the confidence/co-occurrence gate passes.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from doc_extractor.agents.base import BaseAgent
from doc_extractor.exceptions import StageValidationError
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.reviewer import load_lexicons
from doc_extractor.schemas.attribute import Attribute
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.common import Evidence
from doc_extractor.schemas.entity import CanonicalEntity, Entity
from doc_extractor.schemas.mapping import Mapping, RelationshipType
from doc_extractor.schemas.review import ReviewDecision
from doc_extractor.schemas.state import STAGE_ORDER, PipelineState, StageName
from doc_extractor.storage import ids

_RFC2119_RE = re.compile(r"\b(SHALL|MUST|SHOULD|REQUIRED)\b")


def _relationship_type(attr: Attribute) -> RelationshipType:
    text = " ".join(t for t in (attr.binding.evidence, attr.source_text) if t)
    if _RFC2119_RE.search(text):
        return "constrains"
    scope = attr.binding.scope
    if scope in ("table_subject", "nearest_heading", "syntactic"):
        return "has_attribute"
    return "references"


class AttributeMappingAgent(BaseAgent):
    name: ClassVar[StageName] = StageName.attribute_mapping
    requires: ClassVar[tuple[str, ...]] = ("canonical_entities", "review_decisions", "attributes", "chunks")
    produces: ClassVar[tuple[str, ...]] = ("mappings", "unmapped_attribute_ids")

    def validate_input(self, state: PipelineState) -> None:
        idx = STAGE_ORDER.index(self.name)
        prev = STAGE_ORDER[idx - 1]
        rec = (state.get("stages") or {}).get(str(prev))
        if rec is None or rec.status != "succeeded":
            raise StageValidationError(
                str(self.name),
                f"predecessor stage '{prev}' has status {rec.status if rec else 'missing'}",
                phase="input",
            )
        for key in self.requires:
            if key == "attributes":
                if "attributes" not in state:
                    raise StageValidationError(
                        str(self.name), "required state key 'attributes' is missing", phase="input"
                    )
                continue
            value = state.get(key)
            if value is None or (isinstance(value, list | dict | str) and len(value) == 0):
                raise StageValidationError(
                    str(self.name), f"required state key '{key}' is missing or empty", phase="input"
                )

    def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
        attributes: list[Attribute] = list(state.get("attributes", []))
        chunks: list[Chunk] = state["chunks"]
        entities: list[Entity] = list(state.get("entities", []))
        canonical_entities: list[CanonicalEntity] = state["canonical_entities"]
        decisions: list[ReviewDecision] = state["review_decisions"]
        lexicons = load_lexicons(self.cfg)

        decisions_by_id = {d.entity_id: d for d in decisions}
        cents_by_id = {c.canonical_id: c for c in canonical_entities}
        chunks_by_id = {c.chunk_id: c for c in chunks}
        eligible_statuses = {"ACCEPT"} | ({"REVIEW"} if self.cfg.mapping.include_review_entities else set())

        member_to_canonical: dict[str, str] = {}
        for cent in canonical_entities:
            for member_id in cent.member_ids:
                member_to_canonical[member_id] = cent.canonical_id

        entities_by_name: dict[str, list[Entity]] = {}
        for entity in entities:
            entities_by_name.setdefault(entity.entity_name.casefold(), []).append(entity)

        mappings: list[Mapping] = []
        mapped_attribute_ids: set[str] = set()
        min_conf = self.cfg.mapping.min_confidence

        for attr in sorted(attributes, key=lambda a: a.attribute_id):
            bound_name = attr.binding.entity_name
            if not bound_name:
                continue
            original_entity = self._resolve_original_entity(bound_name, entities_by_name, lexicons)
            if original_entity is None:
                trace.count("mapping_unresolved_entity")
                continue
            canonical_id = member_to_canonical.get(original_entity.entity_id)
            if canonical_id is None:
                trace.count("mapping_no_canonical_group")
                continue
            decision = decisions_by_id.get(canonical_id)
            target_cent = cents_by_id.get(canonical_id)
            if decision is None or target_cent is None or decision.validation_status not in eligible_statuses:
                trace.count("mapping_ineligible_entity")
                continue

            rel = _relationship_type(attr)
            confidence = attr.binding.confidence * attr.confidence * decision.overall_score
            if confidence < min_conf:
                trace.count("mapping_below_min_confidence")
                continue
            shares_chunk = attr.source_chunk in target_cent.source_chunks
            names_entity = bool(attr.binding.evidence) and (
                original_entity.entity_name.casefold() in (attr.binding.evidence or "").casefold()
            )
            if not (shares_chunk or names_entity):
                trace.count("mapping_no_cooccurrence")
                continue

            chunk = chunks_by_id.get(attr.source_chunk)
            supporting_evidence: list[Evidence] = []
            seen_ev: set[tuple[str, str]] = set()
            for ev in attr.evidence:
                ev_chunk = chunks_by_id.get(ev.chunk_id)
                if (
                    ev_chunk is not None
                    and ev.text in ev_chunk.source_text
                    and (ev.chunk_id, ev.text) not in seen_ev
                ):
                    supporting_evidence.append(ev)
                    seen_ev.add((ev.chunk_id, ev.text))
            if chunk is not None and attr.binding.evidence and attr.binding.evidence in chunk.source_text:
                key = (attr.source_chunk, attr.binding.evidence)
                if key not in seen_ev:
                    supporting_evidence.append(
                        Evidence(chunk_id=attr.source_chunk, text=attr.binding.evidence)
                    )
                    seen_ev.add(key)
            supporting_evidence.sort(key=lambda e: (e.chunk_id, e.text))

            mapping = Mapping(
                mapping_id=ids.mapping_id(attr.attribute_id, canonical_id, rel),
                attribute_id=attr.attribute_id,
                entity_id=canonical_id,
                relationship_type=rel,
                source_chunk=attr.source_chunk,
                supporting_evidence=supporting_evidence,
                confidence=round(confidence, 6),
            )
            mappings.append(mapping)
            mapped_attribute_ids.add(attr.attribute_id)

        unmapped_ids = sorted({a.attribute_id for a in attributes} - mapped_attribute_ids)
        mappings.sort(key=lambda m: m.mapping_id)
        trace.count("mappings_created", len(mappings))
        trace.count("attributes_unmapped", len(unmapped_ids))
        return {"mappings": mappings, "unmapped_attribute_ids": unmapped_ids}

    def _resolve_original_entity(
        self, bound_name: str, entities_by_name: dict[str, list[Entity]], lexicons: Any
    ) -> Entity | None:
        candidates = entities_by_name.get(bound_name.casefold())
        if not candidates:
            ke = lexicons.known_entity_lookup(bound_name)
            if ke is not None:
                candidates = entities_by_name.get(ke.name.casefold())
        if not candidates:
            return None
        return sorted(candidates, key=lambda e: e.entity_id)[0]

    # ---- validation -----------------------------------------------------------------------
    def validate_output(self, delta: dict[str, Any], state: PipelineState) -> None:
        super().validate_output(delta, state)
        mappings: list[Mapping] = delta["mappings"]
        unmapped: list[str] = delta["unmapped_attribute_ids"]
        decisions_by_id = {d.entity_id: d for d in state["review_decisions"]}
        cents_by_id = {c.canonical_id: c for c in state["canonical_entities"]}
        attr_ids = {a.attribute_id for a in state.get("attributes", [])}
        eligible_statuses = {"ACCEPT"} | ({"REVIEW"} if self.cfg.mapping.include_review_entities else set())

        seen_pairs: set[tuple[str, str]] = set()
        for m in mappings:
            if m.attribute_id not in attr_ids:
                raise StageValidationError(
                    str(self.name), f"mapping references unknown attribute {m.attribute_id}"
                )
            if m.entity_id not in cents_by_id:
                raise StageValidationError(
                    str(self.name), f"mapping references unknown canonical entity {m.entity_id}"
                )
            decision = decisions_by_id.get(m.entity_id)
            if decision is None or decision.validation_status not in eligible_statuses:
                raise StageValidationError(
                    str(self.name), f"mapping {m.mapping_id} targets an ineligible entity {m.entity_id}"
                )
            key = (m.attribute_id, m.relationship_type)
            if key in seen_pairs:
                raise StageValidationError(
                    str(self.name), f"duplicate (attribute_id, relationship_type) {key}"
                )
            seen_pairs.add(key)
        if len(unmapped) != len(set(unmapped)):
            raise StageValidationError(str(self.name), "unmapped_attribute_ids contains duplicates")
