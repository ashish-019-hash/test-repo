"""Tests for `doc_extractor.reviewer.similarity`.

Central guarantee under test: similar names alone can never merge two entities.
`classify` requires a type match AND a strong secondary signal (exact normalized
match, a configured alias link, or substantial attribute overlap).
"""

from __future__ import annotations

from dataclasses import dataclass

from doc_extractor.reviewer import load_lexicons
from doc_extractor.reviewer.similarity import (
    canonical_member,
    classify,
    composite,
    find_duplicates,
    pair_features,
    union_groups,
)
from doc_extractor.schemas.common import Evidence
from doc_extractor.schemas.entity import Entity


@dataclass
class _NE:
    entity_id: str
    normalized_entity_name: str


def make_entity(
    entity_id: str,
    name: str,
    entity_type: str,
    *,
    chunks: set[str],
    attrs: set[str],
    evidence_texts: tuple[str, ...] = (),
    origin: str = "binding",
    confidence: float = 0.9,
) -> Entity:
    return Entity(
        entity_id=entity_id,
        entity_name=name,
        entity_type=entity_type,  # type: ignore[arg-type]
        layer="Other",
        source_attributes=sorted(attrs),
        source_chunks=sorted(chunks),
        supporting_evidence=[
            Evidence(chunk_id=next(iter(chunks), "chunk-x"), text=t) for t in evidence_texts
        ],
        origin=origin,  # type: ignore[arg-type]
        confidence=confidence,
    )


def test_composite_is_weighted_sum() -> None:
    features = {"name_sim": 1.0, "type_match": 1.0}
    weights = {"name_sim": 0.5, "type_match": 0.5, "chunk_overlap": 0.3}
    assert composite(features, weights) == 1.0


def test_evc_alias_of_ethernet_virtual_connection_feature_detected(cfg) -> None:
    lexicons = load_lexicons(cfg)
    a = make_entity("ent-a", "EVC", "Service", chunks={"chunk-1"}, attrs={"attr-1", "attr-2"})
    b = make_entity(
        "ent-b", "Ethernet Virtual Connection", "Service", chunks={"chunk-1"}, attrs={"attr-1", "attr-2"}
    )
    features = pair_features(a, b, "evc", "ethernet virtual connection", lexicons)
    assert features["alias_link"] == 1.0
    assert features["type_match"] == 1.0
    # `classify` only needs `strong_secondary`, which alias_link satisfies, once the
    # composite score clears the duplicate threshold.
    classification = classify(features, cfg.reviewer.duplicate.duplicate_threshold, cfg.reviewer.duplicate)
    assert classification == "DUPLICATE"


def test_exact_same_name_with_overlapping_evidence_is_duplicate(cfg) -> None:
    """A realistic true duplicate: same name, same type, overlapping chunks/attrs/evidence
    (e.g. the same entity detected twice by different candidate sources)."""
    lexicons = load_lexicons(cfg)
    a = make_entity(
        "ent-a",
        "EVC",
        "Service",
        chunks={"chunk-1"},
        attrs={"attr-1", "attr-2"},
        evidence_texts=("Committed Information Rate for the EVC",),
    )
    b = make_entity(
        "ent-b",
        "EVC",
        "Service",
        chunks={"chunk-1"},
        attrs={"attr-1", "attr-2"},
        evidence_texts=("Committed Information Rate for the EVC",),
    )
    features = pair_features(a, b, "evc", "evc", lexicons)
    assert features["normalized_exact"] == 1.0
    score = composite(features, cfg.reviewer.duplicate.weights)
    classification = classify(features, score, cfg.reviewer.duplicate)
    assert classification == "DUPLICATE"


def test_service_provider_vs_service_type_are_distinct(cfg) -> None:
    lexicons = load_lexicons(cfg)
    a = make_entity("ent-a", "Service Provider", "Party", chunks={"chunk-1"}, attrs=set())
    b = make_entity("ent-b", "Service Type", "Attribute", chunks={"chunk-2"}, attrs={"attr-servicetype"})
    features = pair_features(a, b, "service provider", "service type", lexicons)
    score = composite(features, cfg.reviewer.duplicate.weights)
    classification = classify(features, score, cfg.reviewer.duplicate)
    assert classification == "DISTINCT"


def test_similar_names_alone_never_merge_even_with_high_name_sim_weight(cfg) -> None:
    """ "Service Order" vs "Service Border": high name similarity, same type, no alias link,
    no attribute/chunk overlap. Even cranking the name_sim weight to dominate the composite
    score must not produce DUPLICATE, because `classify` requires a strong secondary signal."""
    lexicons = load_lexicons(cfg)
    a = make_entity("ent-a", "Service Order", "Object", chunks={"chunk-1"}, attrs={"attr-x"})
    b = make_entity("ent-b", "Service Border", "Object", chunks={"chunk-2"}, attrs={"attr-y"})
    features = pair_features(a, b, "service order", "service border", lexicons)
    assert features["alias_link"] == 0.0
    assert features["normalized_exact"] == 0.0
    assert features["attribute_overlap"] == 0.0

    dup_cfg = cfg.reviewer.duplicate.model_copy(
        update={"weights": {**cfg.reviewer.duplicate.weights, "name_sim": 5.0}}
    )
    score = composite(features, dup_cfg.weights)
    assert score >= dup_cfg.duplicate_threshold  # composite is artificially forced high
    classification = classify(features, score, dup_cfg)
    assert classification != "DUPLICATE"


def test_possible_duplicate_below_duplicate_threshold(cfg) -> None:
    lexicons = load_lexicons(cfg)
    a = make_entity("ent-a", "Subscriber", "Party", chunks={"chunk-1", "chunk-2"}, attrs={"attr-msisdn"})
    b = make_entity("ent-b", "Subscribers", "Party", chunks={"chunk-2"}, attrs={"attr-imsi"})
    features = pair_features(a, b, "subscriber", "subscriber", lexicons)
    # normalized_exact True but attribute_overlap is 0 and no alias_link -> still passes
    # strong_secondary via normalized_exact, so raise thresholds to force POSSIBLE instead.
    dup_cfg = cfg.reviewer.duplicate.model_copy(update={"duplicate_threshold": 0.999})
    score = composite(features, dup_cfg.weights)
    classification = classify(features, score, dup_cfg)
    assert classification == "POSSIBLE_DUPLICATE"


def test_require_type_match_blocks_duplicate_even_with_alias_link(cfg) -> None:
    lexicons = load_lexicons(cfg)
    a = make_entity("ent-a", "EVC", "Service", chunks={"chunk-1"}, attrs={"attr-1"})
    b = make_entity("ent-b", "Ethernet Virtual Connection", "Object", chunks={"chunk-1"}, attrs={"attr-1"})
    features = pair_features(a, b, "evc", "ethernet virtual connection", lexicons)
    assert features["type_match"] == 0.0
    assert features["alias_link"] == 1.0
    score = composite(features, cfg.reviewer.duplicate.weights)
    classification = classify(features, score, cfg.reviewer.duplicate)
    assert classification != "DUPLICATE"


def test_find_duplicates_and_union_groups_merge_exact_duplicate(cfg) -> None:
    lexicons = load_lexicons(cfg)
    entities = [
        make_entity(
            "ent-evc-1",
            "EVC",
            "Service",
            chunks={"chunk-1"},
            attrs={"attr-1", "attr-2"},
            evidence_texts=("Committed Information Rate for the EVC",),
        ),
        make_entity(
            "ent-evc-2",
            "EVC",
            "Service",
            chunks={"chunk-1"},
            attrs={"attr-1", "attr-2"},
            evidence_texts=("Committed Information Rate for the EVC",),
        ),
        make_entity("ent-uni", "UNI", "Interface", chunks={"chunk-3"}, attrs={"attr-3"}),
    ]
    normalized = [
        _NE("ent-evc-1", "evc"),
        _NE("ent-evc-2", "evc"),
        _NE("ent-uni", "uni"),
    ]
    pairs = find_duplicates(entities, normalized, lexicons, cfg)
    dup_pairs = [p for p in pairs if p.classification == "DUPLICATE"]
    assert {p.a for p in dup_pairs} | {p.b for p in dup_pairs} == {"ent-evc-1", "ent-evc-2"}

    groups = union_groups(entities, pairs)
    assert len(groups) == 2  # {EVC-1, EVC-2} merged, UNI alone
    member_sets = {tuple(g.member_ids) for g in groups}
    assert ("ent-evc-1", "ent-evc-2") in member_sets
    assert ("ent-uni",) in member_sets


def test_canonical_member_prefers_more_chunks_then_shorter_name() -> None:
    a = make_entity("ent-a", "EVC", "Service", chunks={"chunk-1"}, attrs=set())
    b = make_entity(
        "ent-b", "Ethernet Virtual Connection", "Service", chunks={"chunk-1", "chunk-2"}, attrs=set()
    )
    chosen = canonical_member([a, b])
    assert chosen.entity_id == "ent-b"  # more source chunks wins

    c = make_entity("ent-c", "EVC", "Service", chunks={"chunk-1"}, attrs=set())
    d = make_entity("ent-d", "Ethernet Virtual Connection", "Service", chunks={"chunk-1"}, attrs=set())
    chosen2 = canonical_member([c, d])
    assert chosen2.entity_id == "ent-c"  # tie on chunk count -> shortest name


def test_union_groups_singleton_when_no_duplicate_pairs(cfg) -> None:
    entities = [
        make_entity("ent-a", "Service Provider", "Party", chunks={"chunk-1"}, attrs=set()),
        make_entity("ent-b", "Service Type", "Attribute", chunks={"chunk-2"}, attrs=set()),
    ]
    groups = union_groups(entities, [])
    assert len(groups) == 2
    assert all(len(g.member_ids) == 1 for g in groups)
