"""Tests for `doc_extractor.reviewer.quality` — the eight deterministic quality scorers."""

from __future__ import annotations

from doc_extractor.config import load_config
from doc_extractor.reviewer import load_lexicons
from doc_extractor.reviewer.quality import finalize_decision, score_criteria, score_entity
from doc_extractor.schemas.attribute import Attribute
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.common import Evidence
from doc_extractor.schemas.entity import CanonicalEntity, Entity

DOC_ID = "doc-abc123abc123abc1"


def make_chunk(chunk_id: str, text: str, heading_path: list[str]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=DOC_ID,
        page_number=1,
        page_end=1,
        source_text=text,
        block_ids=["blk-x"],
        heading_path=heading_path,
        token_count=max(1, len(text.split())),
        strategy="heading_aware",
    )


def make_cent(
    canonical_id: str,
    canonical_name: str,
    *,
    entity_type: str = "Service",
    layer: str = "CFS",
    member_ids: list[str] | None = None,
    source_attributes: list[str] | None = None,
    source_chunks: list[str] | None = None,
    supporting_evidence: list[Evidence] | None = None,
    confidence: float = 0.9,
) -> CanonicalEntity:
    return CanonicalEntity(
        canonical_id=canonical_id,
        canonical_name=canonical_name,
        normalized_name=canonical_name.casefold(),
        entity_type=entity_type,  # type: ignore[arg-type]
        layer=layer,  # type: ignore[arg-type]
        member_ids=member_ids or [f"ent-{canonical_id}"],
        source_attributes=source_attributes or [],
        source_chunks=source_chunks or [],
        supporting_evidence=supporting_evidence or [],
        confidence=confidence,
    )


CHUNK_EVC = make_chunk(
    "chunk-evc",
    "CIR | Integer | Mbps | M | Committed Information Rate for the EVC",
    ["Spec", "4 Service Attributes", "4.3 EVC Service Attributes"],
)


def test_zero_grounding_forces_reject(cfg) -> None:
    lexicons = load_lexicons(cfg)
    cent = make_cent(
        "cent-x",
        "EVC",
        source_attributes=["attr-cir"],
        source_chunks=[CHUNK_EVC.chunk_id],
        supporting_evidence=[],  # no evidence at all -> source_grounding == 0.0
    )
    attr = Attribute.model_construct(
        attribute_id="attr-cir",
        display_name="CIR",
        attribute_name="cir",
    )
    criteria = score_criteria(
        cent,
        [cent],
        entities_by_id={},
        attributes_by_id={"attr-cir": attr},
        chunks_by_id={CHUNK_EVC.chunk_id: CHUNK_EVC},
        lexicons=lexicons,
        pair_composites={},
    )
    grounding = next(c for c in criteria if c.criterion == "source_grounding")
    assert grounding.score == 0.0

    decision = finalize_decision(cent, criteria, cfg)
    assert decision.validation_status == "REJECT"
    assert "no_verified_evidence" in decision.issues


def test_zero_grounding_forces_reject_even_with_high_overall(cfg) -> None:
    """Force every other criterion to 1.0 by cranking the quality weights toward zero for
    every criterion except source_grounding: overall would be near 0 anyway. Instead we
    directly exercise `finalize_decision`'s override by constructing near-perfect criteria
    with only source_grounding at 0."""
    from doc_extractor.schemas.review import QUALITY_CRITERIA, CriterionScore

    cent = make_cent("cent-y", "EVC", source_attributes=["attr-1"], source_chunks=["chunk-1"])
    criteria = [
        CriterionScore(criterion=c, score=(0.0 if c == "source_grounding" else 1.0), reason="test")
        for c in QUALITY_CRITERIA
    ]
    decision = finalize_decision(cent, criteria, cfg)
    assert decision.overall_score > cfg.reviewer.quality.accept  # would ACCEPT on score alone
    assert decision.validation_status == "REJECT"


def test_accept_for_well_grounded_lexicon_entity(cfg) -> None:
    lexicons = load_lexicons(cfg)
    evidence = [Evidence(chunk_id=CHUNK_EVC.chunk_id, text="Committed Information Rate for the EVC")]
    cent = make_cent(
        "cent-evc",
        "EVC",
        entity_type="Service",
        layer="CFS",
        source_attributes=["attr-cir"],
        source_chunks=[CHUNK_EVC.chunk_id],
        supporting_evidence=evidence,
    )
    attr = Attribute.model_construct(attribute_id="attr-cir", display_name="CIR", attribute_name="cir")
    entity = Entity(
        entity_id="ent-cent-evc",
        entity_name="EVC",
        entity_type="Service",  # type: ignore[arg-type]
        layer="CFS",
        source_attributes=["attr-cir"],
        source_chunks=[CHUNK_EVC.chunk_id],
        supporting_evidence=evidence,
        origin="binding",  # type: ignore[arg-type]
        confidence=0.9,
    )
    decision = score_entity(
        cent,
        [cent],
        entities_by_id={entity.entity_id: entity},
        attributes_by_id={"attr-cir": attr},
        chunks_by_id={CHUNK_EVC.chunk_id: CHUNK_EVC},
        lexicons=lexicons,
        cfg=cfg,
        pair_composites={},
    )
    assert decision.validation_status == "ACCEPT"
    assert decision.overall_score >= cfg.reviewer.quality.accept


def test_generic_term_penalizes_specificity(cfg) -> None:
    lexicons = load_lexicons(cfg)
    cent = make_cent("cent-generic", "System", entity_type="Object", layer="Other")
    criteria = score_criteria(
        cent,
        [cent],
        entities_by_id={},
        attributes_by_id={},
        chunks_by_id={},
        lexicons=lexicons,
        pair_composites={},
    )
    specificity = next(c for c in criteria if c.criterion == "specificity")
    assert specificity.score < 1.0


def test_completeness_checks_all_four_dimensions(cfg) -> None:
    incomplete = make_cent("cent-i", "Widget", entity_type="Object", layer="Other")
    complete = make_cent(
        "cent-c",
        "EVC",
        entity_type="Service",
        layer="CFS",
        source_attributes=["attr-1"],
        supporting_evidence=[Evidence(chunk_id="chunk-1", text="x")],
    )
    lexicons = load_lexicons(cfg)
    incomplete_score = score_criteria(incomplete, [incomplete], {}, {}, {}, lexicons, {})
    complete_score = score_criteria(complete, [complete], {}, {}, {}, lexicons, {})
    inc = next(c for c in incomplete_score if c.criterion == "completeness")
    comp = next(c for c in complete_score if c.criterion == "completeness")
    assert inc.score == 0.0
    assert comp.score == 1.0


def test_disambiguation_potential_penalizes_shared_names(cfg) -> None:
    lexicons = load_lexicons(cfg)
    a = make_cent("cent-a", "Order", source_attributes=["attr-1"])
    b = make_cent("cent-b", "Order", source_attributes=["attr-2"])
    scores_a = score_criteria(a, [a, b], {}, {}, {}, lexicons, {})
    disambig_a = next(c for c in scores_a if c.criterion == "disambiguation_potential")
    assert disambig_a.score < 1.0


def test_thresholds_are_configurable(cfg) -> None:
    from doc_extractor.schemas.review import QUALITY_CRITERIA, CriterionScore

    cent = make_cent("cent-evc", "EVC", source_attributes=["attr-cir"], source_chunks=[CHUNK_EVC.chunk_id])
    criteria = [CriterionScore(criterion=c, score=0.8, reason="uniform") for c in QUALITY_CRITERIA]

    default_decision = finalize_decision(cent, criteria, cfg)
    assert default_decision.overall_score == 0.8
    assert default_decision.validation_status == "ACCEPT"  # 0.8 >= default accept (0.75)

    stricter = load_config(env={"LLM_PROVIDER": "rules", "DOC_EXTRACTOR__reviewer__quality__accept": "0.95"})
    stricter_decision = finalize_decision(cent, criteria, stricter)
    assert stricter_decision.validation_status == "REVIEW"  # 0.8 < 0.95 accept but >= review
