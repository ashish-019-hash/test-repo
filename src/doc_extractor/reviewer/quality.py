"""The eight deterministic quality criterion scorers for canonical entities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.attribute import Attribute
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.entity import CanonicalEntity, Entity
from doc_extractor.schemas.review import CriterionScore, ReviewDecision


def _score_specificity(cent: CanonicalEntity, lexicons: Any) -> CriterionScore:
    tokens = cent.canonical_name.split()
    generic_hits = sum(1 for t in tokens if t.casefold() in lexicons.generic_terms)
    score = 1.0 - min(1.0, 0.5 * generic_hits)
    bonus_reasons = []
    if len(tokens) >= 2:
        score = min(1.0, score + 0.1)
        bonus_reasons.append("multi-word")
    if cent.canonical_name.replace(" ", "").isupper() and len(cent.canonical_name) <= 6:
        score = min(1.0, score + 0.1)
        bonus_reasons.append("acronym")
    score = max(0.0, min(1.0, score))
    reason = f"{generic_hits} generic term(s)" + (
        f"; bonus: {', '.join(bonus_reasons)}" if bonus_reasons else ""
    )
    return CriterionScore(criterion="specificity", score=score, reason=reason)


def _score_uniqueness(cent: CanonicalEntity, pair_composites: dict[tuple[str, str], float]) -> CriterionScore:
    sims = [c for (x, y), c in pair_composites.items() if cent.canonical_id in (x, y)]
    max_sim = max(sims, default=0.0)
    score = max(0.0, 1.0 - max_sim)
    reason = f"max composite similarity to another canonical entity = {max_sim:.2f}"
    return CriterionScore(criterion="uniqueness", score=score, reason=reason)


def _score_real_world_correspondence(
    cent: CanonicalEntity, entities_by_id: dict[str, Entity], lexicons: Any
) -> CriterionScore:
    ke = lexicons.known_entity_lookup(cent.canonical_name)
    in_sid_abes = any(name.casefold() == cent.canonical_name.casefold() for name in lexicons.sid_abes)
    if ke is not None or in_sid_abes:
        return CriterionScore(
            criterion="real_world_correspondence", score=1.0, reason="matches known_entities/sid_abes"
        )
    if len(cent.source_attributes) >= 2:
        return CriterionScore(
            criterion="real_world_correspondence",
            score=0.7,
            reason=f"bound by {len(cent.source_attributes)} attributes",
        )
    origins = {entities_by_id[m].origin for m in cent.member_ids if m in entities_by_id}
    if origins and origins == {"llm"}:
        return CriterionScore(
            criterion="real_world_correspondence", score=0.4, reason="llm-only origin, unverified"
        )
    return CriterionScore(
        criterion="real_world_correspondence", score=0.5, reason="no lexicon or strong attribute support"
    )


def _score_contextual_relevance(
    cent: CanonicalEntity, attributes_by_id: dict[str, Attribute], chunks_by_id: dict[str, Chunk]
) -> CriterionScore:
    if not cent.source_chunks:
        return CriterionScore(criterion="contextual_relevance", score=0.0, reason="no source chunks")
    mention_terms = {cent.canonical_name.casefold(), cent.normalized_name.casefold()}
    for attr_id in cent.source_attributes:
        attr = attributes_by_id.get(attr_id)
        if attr is not None:
            mention_terms.add(attr.display_name.casefold())
            mention_terms.add(attr.attribute_name.casefold())
    hits = 0
    for chunk_id in cent.source_chunks:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        headings = " ".join(chunk.heading_path).casefold()
        if any(term and term in headings for term in mention_terms):
            hits += 1
    score = hits / len(cent.source_chunks)
    return CriterionScore(
        criterion="contextual_relevance",
        score=score,
        reason=f"{hits}/{len(cent.source_chunks)} source chunk heading(s) mention the entity",
    )


def _score_completeness(cent: CanonicalEntity) -> CriterionScore:
    checks = [
        bool(cent.entity_type) and cent.entity_type != "Object",
        cent.layer != "Other",
        len(cent.source_attributes) >= 1,
        len(cent.supporting_evidence) >= 1,
    ]
    score = min(1.0, sum(1 for c in checks if c) / 4)
    return CriterionScore(
        criterion="completeness", score=score, reason=f"{sum(checks)}/4 completeness checks passed"
    )


def _score_source_grounding(cent: CanonicalEntity, chunks_by_id: dict[str, Chunk]) -> CriterionScore:
    total = len(cent.supporting_evidence)
    if total == 0:
        return CriterionScore(criterion="source_grounding", score=0.0, reason="no supporting evidence")
    verified = 0
    for ev in cent.supporting_evidence:
        chunk = chunks_by_id.get(ev.chunk_id)
        if chunk is not None and ev.text in chunk.source_text:
            verified += 1
    score = verified / total
    return CriterionScore(
        criterion="source_grounding", score=score, reason=f"{verified}/{total} evidence strings verified"
    )


def _score_entity_type_correctness(cent: CanonicalEntity, lexicons: Any) -> CriterionScore:
    ke = lexicons.known_entity_lookup(cent.canonical_name)
    if ke is None:
        return CriterionScore(
            criterion="entity_type_correctness", score=0.5, reason="type/layer not in known_entities"
        )
    if ke.layer == cent.layer:
        return CriterionScore(
            criterion="entity_type_correctness",
            score=1.0,
            reason=f"layer matches known_entities ({ke.layer})",
        )
    return CriterionScore(
        criterion="entity_type_correctness",
        score=0.0,
        reason=f"layer {cent.layer} contradicts known_entities layer {ke.layer}",
    )


def _score_disambiguation_potential(
    cent: CanonicalEntity, all_cents: Sequence[CanonicalEntity]
) -> CriterionScore:
    name_cf = cent.canonical_name.casefold()
    same_name_others = [
        c for c in all_cents if c.canonical_id != cent.canonical_id and c.canonical_name.casefold() == name_cf
    ]
    has_support = len(cent.source_attributes) >= 1
    if not same_name_others:
        score = 1.0 if has_support else 0.6
        reason = "unique name in document" + (" with attribute support" if has_support else "")
    else:
        score = max(0.0, 1.0 - 0.3 * len(same_name_others))
        reason = f"{len(same_name_others)} other canonical entity(ies) share this name"
    return CriterionScore(criterion="disambiguation_potential", score=score, reason=reason)


def score_criteria(
    cent: CanonicalEntity,
    all_cents: Sequence[CanonicalEntity],
    entities_by_id: dict[str, Entity],
    attributes_by_id: dict[str, Attribute],
    chunks_by_id: dict[str, Chunk],
    lexicons: Any,
    pair_composites: dict[tuple[str, str], float],
) -> list[CriterionScore]:
    """Compute the eight deterministic criterion scores (rule-based, no LLM)."""
    return [
        _score_specificity(cent, lexicons),
        _score_uniqueness(cent, pair_composites),
        _score_real_world_correspondence(cent, entities_by_id, lexicons),
        _score_contextual_relevance(cent, attributes_by_id, chunks_by_id),
        _score_completeness(cent),
        _score_source_grounding(cent, chunks_by_id),
        _score_entity_type_correctness(cent, lexicons),
        _score_disambiguation_potential(cent, all_cents),
    ]


def finalize_decision(
    cent: CanonicalEntity, criteria: Sequence[CriterionScore], cfg: AppConfig
) -> ReviewDecision:
    """Combine criterion scores (rule-based, possibly LLM-adjusted) into a ReviewDecision."""
    scores = {c.criterion: c.score for c in criteria}
    weights = cfg.reviewer.quality.weights
    overall = sum(weights.get(c.criterion, 0.0) * c.score for c in criteria)
    confidence = min(cent.confidence, overall)

    status: Literal["ACCEPT", "REVIEW", "REJECT"]
    source_grounding_zero = scores["source_grounding"] == 0.0
    if source_grounding_zero:
        status = "REJECT"
    elif overall >= cfg.reviewer.quality.accept:
        status = "ACCEPT"
    elif overall >= cfg.reviewer.quality.review:
        status = "REVIEW"
    else:
        status = "REJECT"

    issues = [f"{c.criterion}_low" for c in criteria if c.score < 0.5]
    if source_grounding_zero and "no_verified_evidence" not in issues:
        issues.append("no_verified_evidence")
    issues = sorted(set(issues))

    reason = f"{cent.canonical_name}: overall={overall:.2f} -> {status}"
    if issues:
        reason += f" (issues: {', '.join(issues)})"

    return ReviewDecision(
        entity_id=cent.canonical_id,
        criterion_scores=list(criteria),
        overall_score=max(0.0, min(1.0, overall)),
        confidence=max(0.0, min(1.0, confidence)),
        validation_status=status,
        issues=issues,
        reason=reason,
    )


def score_entity(
    cent: CanonicalEntity,
    all_cents: Sequence[CanonicalEntity],
    entities_by_id: dict[str, Entity],
    attributes_by_id: dict[str, Attribute],
    chunks_by_id: dict[str, Chunk],
    lexicons: Any,
    cfg: AppConfig,
    pair_composites: dict[tuple[str, str], float],
) -> ReviewDecision:
    """Rule-based-only convenience wrapper: score_criteria + finalize_decision."""
    criteria = score_criteria(
        cent, all_cents, entities_by_id, attributes_by_id, chunks_by_id, lexicons, pair_composites
    )
    return finalize_decision(cent, criteria, cfg)
