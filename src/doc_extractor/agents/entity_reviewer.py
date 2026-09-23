"""Agent 7: EntityReviewerAgent.

Duplicate detection (`reviewer/similarity.py`) followed by eight-criterion quality
validation (`reviewer/quality.py`). Builds one CanonicalEntity per duplicate group
(singletons included) and exactly one ReviewDecision per canonical entity.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, ClassVar

from doc_extractor.agents.base import BaseAgent
from doc_extractor.exceptions import StageValidationError
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.reviewer import load_lexicons
from doc_extractor.reviewer.quality import finalize_decision, score_criteria
from doc_extractor.reviewer.similarity import (
    canonical_member,
    composite,
    find_duplicates,
    pair_features,
    union_groups,
)
from doc_extractor.schemas.common import Evidence
from doc_extractor.schemas.entity import CanonicalEntity, Entity
from doc_extractor.schemas.review import CriterionScore, ReviewDecision
from doc_extractor.schemas.state import STAGE_ORDER, HumanReviewItem, PipelineState, StageName
from doc_extractor.storage import ids


class EntityReviewerAgent(BaseAgent):
    name: ClassVar[StageName] = StageName.entity_reviewer
    requires: ClassVar[tuple[str, ...]] = ("entities", "normalized_entities", "attributes", "chunks")
    produces: ClassVar[tuple[str, ...]] = ("duplicate_groups", "canonical_entities", "review_decisions")

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
        entities = state["entities"]
        normalized = state["normalized_entities"]
        chunks = state["chunks"]
        attributes = list(state.get("attributes", []))
        lexicons = load_lexicons(self.cfg)

        pairs = find_duplicates(entities, normalized, lexicons, self.cfg)
        groups = union_groups(entities, pairs)

        entities_by_id = {e.entity_id: e for e in entities}
        normalized_by_id = {n.entity_id: n for n in normalized}
        attributes_by_id = {a.attribute_id: a for a in attributes}
        chunks_by_id = {c.chunk_id: c for c in chunks}

        cents = self._build_canonical_entities(groups, entities_by_id, normalized_by_id)
        pair_composites = self._pairwise_composites(cents, lexicons)

        decisions: list[ReviewDecision] = []
        for cent in cents:
            criteria = score_criteria(
                cent, cents, entities_by_id, attributes_by_id, chunks_by_id, lexicons, pair_composites
            )
            if self.provider.name == "azure":
                criteria = self._apply_llm_adjustment(cent, criteria, trace)
            decisions.append(finalize_decision(cent, criteria, self.cfg))
        decisions.sort(key=lambda d: d.entity_id)

        review_items = self._build_review_items(pairs, decisions)

        trace.count("duplicate_groups", len(groups))
        trace.count("canonical_entities", len(cents))
        trace.count("review_decisions", len(decisions))
        return {
            "duplicate_groups": groups,
            "canonical_entities": cents,
            "review_decisions": decisions,
            "human_review_queue": review_items,
        }

    # ---- helpers ------------------------------------------------------------------------
    def _build_canonical_entities(
        self,
        groups: list[Any],
        entities_by_id: dict[str, Entity],
        normalized_by_id: dict[str, Any],
    ) -> list[CanonicalEntity]:
        cents: list[CanonicalEntity] = []
        for group in groups:
            members = [entities_by_id[m] for m in group.member_ids]
            cm = canonical_member(members)
            ne = normalized_by_id[cm.entity_id]
            source_attributes = sorted({a for m in members for a in m.source_attributes})
            source_chunks = sorted({c for m in members for c in m.source_chunks})
            evidence_map: dict[tuple[str, str], Evidence] = {}
            for m in members:
                for ev in m.supporting_evidence:
                    evidence_map[(ev.chunk_id, ev.text)] = ev
            confidence = max(m.confidence for m in members)
            cents.append(
                CanonicalEntity(
                    canonical_id=group.canonical_id,
                    canonical_name=cm.entity_name,
                    normalized_name=ne.normalized_entity_name,
                    entity_type=cm.entity_type,
                    layer=cm.layer,
                    member_ids=sorted(group.member_ids),
                    source_attributes=source_attributes,
                    source_chunks=source_chunks,
                    supporting_evidence=sorted(evidence_map.values(), key=lambda e: (e.chunk_id, e.text)),
                    confidence=round(confidence, 6),
                )
            )
        cents.sort(key=lambda c: c.canonical_id)
        return cents

    def _pairwise_composites(
        self, cents: list[CanonicalEntity], lexicons: Any
    ) -> dict[tuple[str, str], float]:
        weights = self.cfg.reviewer.duplicate.weights
        out: dict[tuple[str, str], float] = {}
        for a, b in combinations(cents, 2):
            features = pair_features(a, b, a.normalized_name, b.normalized_name, lexicons)
            score = composite(features, weights)
            lo, hi = sorted((a.canonical_id, b.canonical_id))
            out[(lo, hi)] = score
        return out

    def _apply_llm_adjustment(
        self, cent: CanonicalEntity, criteria: list[CriterionScore], trace: TraceCollector
    ) -> list[CriterionScore]:
        from doc_extractor.llm import tasks as llm_tasks  # lazy: azure-mode only

        task = _resolve_llm_task(llm_tasks, "entity_quality")
        payload = {
            "canonical_id": cent.canonical_id,
            "canonical_name": cent.canonical_name,
            "entity_type": cent.entity_type,
            "layer": cent.layer,
            "supporting_evidence": [e.text for e in cent.supporting_evidence],
        }
        result = self.provider.call(task, payload)
        trace.record_llm(result.cache_hit)
        max_adj = self.cfg.reviewer.quality.llm_adjustment_max
        by_name = {c.criterion: c for c in criteria}
        for crit_name in ("specificity", "real_world_correspondence"):
            delta = getattr(result.response, f"{crit_name}_delta", 0.0) or 0.0
            delta = max(-max_adj, min(max_adj, delta))
            if delta == 0.0:
                continue
            llm_reason = getattr(result.response, f"{crit_name}_reason", "") or ""
            cs = by_name[crit_name]
            new_score = max(0.0, min(1.0, cs.score + delta))
            sign = "+" if delta >= 0 else ""
            by_name[crit_name] = CriterionScore(
                criterion=crit_name,
                score=new_score,
                reason=f"{cs.reason}; llm_adjustment={sign}{delta:.2f} ({llm_reason})",
            )
        return [by_name[c.criterion] for c in criteria]

    def _build_review_items(self, pairs: list[Any], decisions: list[ReviewDecision]) -> list[HumanReviewItem]:
        items: list[HumanReviewItem] = []
        for pair in pairs:
            if pair.classification == "POSSIBLE_DUPLICATE":
                items.append(
                    HumanReviewItem(
                        item_id=ids.review_item_id("possible_duplicate", [pair.a, pair.b]),
                        kind="possible_duplicate",
                        ref_ids=[pair.a, pair.b],
                        score=pair.composite,
                        reason=pair.reason,
                        payload={},
                    )
                )
        for decision in decisions:
            if decision.validation_status == "REVIEW":
                items.append(
                    HumanReviewItem(
                        item_id=ids.review_item_id("entity_review", [decision.entity_id]),
                        kind="entity_review",
                        ref_ids=[decision.entity_id],
                        score=decision.overall_score,
                        reason=decision.reason,
                        payload={},
                    )
                )
        items.sort(key=lambda i: i.item_id)
        return items

    # ---- validation -----------------------------------------------------------------------
    def validate_output(self, delta: dict[str, Any], state: PipelineState) -> None:
        super().validate_output(delta, state)
        groups = delta["duplicate_groups"]
        cents: list[CanonicalEntity] = delta["canonical_entities"]
        decisions: list[ReviewDecision] = delta["review_decisions"]
        entity_ids = {e.entity_id for e in state["entities"]}

        seen_members: set[str] = set()
        for group in groups:
            for m in group.member_ids:
                if m not in entity_ids:
                    raise StageValidationError(str(self.name), f"group references unknown entity {m}")
                if m in seen_members:
                    raise StageValidationError(str(self.name), f"entity {m} belongs to more than one group")
                seen_members.add(m)
        if seen_members != entity_ids:
            raise StageValidationError(
                str(self.name), "not every original entity belongs to exactly one group"
            )

        canonical_ids = [c.canonical_id for c in cents]
        if len(canonical_ids) != len(set(canonical_ids)):
            raise StageValidationError(str(self.name), "duplicate canonical_id in canonical_entities")
        group_ids = {g.canonical_id for g in groups}
        if set(canonical_ids) != group_ids:
            raise StageValidationError(str(self.name), "canonical_entities do not match duplicate_groups")

        decision_ids = [d.entity_id for d in decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise StageValidationError(str(self.name), "duplicate entity_id in review_decisions")
        if set(decision_ids) != set(canonical_ids):
            raise StageValidationError(str(self.name), "review_decisions do not match canonical_entities 1:1")


def _resolve_llm_task(module: Any, name: str) -> Any:
    const_name = name.upper()
    if hasattr(module, const_name):
        return getattr(module, const_name)
    if hasattr(module, "TASKS"):
        return module.TASKS[name]
    raise AttributeError(f"LLM task '{name}' not found in doc_extractor.llm.tasks")
