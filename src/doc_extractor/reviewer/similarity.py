"""Pairwise entity similarity features, composite score, duplicate classification.

Similar names alone can never merge two entities into one: `classify` requires a
type match and a strong secondary signal (exact normalized match, a configured
alias link, or substantial attribute overlap) before returning `DUPLICATE`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any

from rapidfuzz import fuzz

from doc_extractor.config.models import DuplicateConfig
from doc_extractor.schemas.entity import Entity
from doc_extractor.schemas.review import DuplicateClass, DuplicateGroup, DuplicatePair
from doc_extractor.storage import ids


def _record_name(x: Any) -> str:
    name = getattr(x, "entity_name", None)
    return name if name is not None else x.canonical_name


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _shingles(text: str) -> set[str]:
    t = text.casefold()
    if len(t) < 3:
        return {t} if t else set()
    return {t[i : i + 3] for i in range(len(t) - 2)}


def _evidence_shingles(record: Any) -> set[str]:
    shingles: set[str] = set()
    for ev in getattr(record, "supporting_evidence", []):
        shingles |= _shingles(ev.text)
    return shingles


def _is_alias_link(name_a: str, name_b: str, na: str, nb: str, lexicons: Any) -> bool:
    candidates_a = {name_a.casefold(), na.casefold()}
    candidates_b = {name_b.casefold(), nb.casefold()}

    def _linked(alias_map: Iterable[tuple[str, Iterable[str]]]) -> bool:
        for canonical, alist in alias_map:
            canon_cf = canonical.casefold()
            alias_cfs = {al.casefold() for al in alist}
            if canon_cf in candidates_a and candidates_b & alias_cfs:
                return True
            if canon_cf in candidates_b and candidates_a & alias_cfs:
                return True
        return False

    if _linked(lexicons.aliases.items()):
        return True
    return _linked((name, ke.aliases) for name, ke in lexicons.known_entities.items())


def pair_features(a: Any, b: Any, na: str, nb: str, lexicons: Any) -> dict[str, float]:
    """Compute the seven [0,1] similarity features for a candidate duplicate pair.

    `a`, `b` are `Entity` (or duck-typed `CanonicalEntity`) records; `na`, `nb` are their
    already-normalized names.
    """
    name_a, name_b = _record_name(a), _record_name(b)
    name_sim = fuzz.token_set_ratio(name_a, name_b) / 100.0
    normalized_exact = 1.0 if na == nb else 0.0
    type_match = 1.0 if a.entity_type == b.entity_type else 0.0
    chunk_overlap = _jaccard(a.source_chunks, b.source_chunks)
    attribute_overlap = _jaccard(a.source_attributes, b.source_attributes)
    evidence_overlap = _jaccard(_evidence_shingles(a), _evidence_shingles(b))
    alias_link = 1.0 if _is_alias_link(name_a, name_b, na, nb, lexicons) else 0.0
    return {
        "name_sim": name_sim,
        "normalized_exact": normalized_exact,
        "type_match": type_match,
        "chunk_overlap": chunk_overlap,
        "attribute_overlap": attribute_overlap,
        "evidence_overlap": evidence_overlap,
        "alias_link": alias_link,
    }


def composite(features: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * features.get(k, 0.0) for k in weights)


def classify(features: dict[str, float], composite_score: float, cfg: DuplicateConfig) -> DuplicateClass:
    type_ok = features.get("type_match", 0.0) == 1.0 or not cfg.require_type_match
    strong_secondary = (
        features.get("normalized_exact", 0.0) == 1.0
        or features.get("alias_link", 0.0) == 1.0
        or features.get("attribute_overlap", 0.0) >= 0.5
    )
    if composite_score >= cfg.duplicate_threshold and type_ok and strong_secondary:
        return "DUPLICATE"
    if composite_score >= cfg.possible_threshold:
        return "POSSIBLE_DUPLICATE"
    return "DISTINCT"


def _reason(features: dict[str, float], composite_score: float, classification: str) -> str:
    parts = [f"composite={composite_score:.3f}"]
    for k in ("type_match", "normalized_exact", "alias_link", "attribute_overlap", "name_sim"):
        parts.append(f"{k}={features.get(k, 0.0):.2f}")
    return f"{classification}: " + ", ".join(parts)


def _blocking_key(entity: Entity, normalized_name: str) -> tuple[str, str]:
    return (normalized_name[:1].casefold() if normalized_name else "", entity.entity_type)


def find_duplicates(
    entities: Sequence[Entity],
    normalized: Sequence[Any],
    lexicons: Any,
    cfg: Any,
) -> list[DuplicatePair]:
    """All non-DISTINCT pairs, blocked by (first letter of normalized name OR type)."""
    dup_cfg: DuplicateConfig = cfg.reviewer.duplicate
    normalized_by_id = {ne.entity_id: ne.normalized_entity_name for ne in normalized}
    ordered = sorted(entities, key=lambda e: e.entity_id)
    pairs: list[DuplicatePair] = []
    for a, b in combinations(ordered, 2):
        na = normalized_by_id.get(a.entity_id, a.entity_name.casefold())
        nb = normalized_by_id.get(b.entity_id, b.entity_name.casefold())
        same_letter = bool(na) and bool(nb) and na[:1] == nb[:1]
        same_type = a.entity_type == b.entity_type
        if not (same_letter or same_type):
            continue
        features = pair_features(a, b, na, nb, lexicons)
        score = composite(features, dup_cfg.weights)
        classification = classify(features, score, dup_cfg)
        if classification == "DISTINCT":
            continue
        lo, hi = (a.entity_id, b.entity_id) if a.entity_id < b.entity_id else (b.entity_id, a.entity_id)
        pairs.append(
            DuplicatePair(
                a=lo,
                b=hi,
                features=features,
                composite=score,
                classification=classification,
                reason=_reason(features, score, classification),
            )
        )
    return pairs


def canonical_member(members: Sequence[Entity]) -> Entity:
    """Most source chunks; tie -> shortest name; tie -> lexicographic name."""
    return min(members, key=lambda e: (-len(e.source_chunks), len(e.entity_name), e.entity_name))


def union_groups(entities: Sequence[Entity], pairs: Sequence[DuplicatePair]) -> list[DuplicateGroup]:
    """Union-find over DUPLICATE pairs only. Every entity ends in exactly one group."""
    parent: dict[str, str] = {e.entity_id: e.entity_id for e in entities}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rx > ry:
            rx, ry = ry, rx
        parent[ry] = rx

    for pair in pairs:
        if pair.classification == "DUPLICATE":
            union(pair.a, pair.b)

    entities_by_id = {e.entity_id: e for e in entities}
    clusters: dict[str, list[str]] = {}
    for e in entities:
        root = find(e.entity_id)
        clusters.setdefault(root, []).append(e.entity_id)

    groups: list[DuplicateGroup] = []
    for member_ids in clusters.values():
        members = [entities_by_id[m] for m in member_ids]
        cm = canonical_member(members)
        sorted_members = sorted(member_ids)
        canon_id = ids.canonical_id(sorted_members)
        group_pairs = [
            p for p in pairs if p.classification != "DISTINCT" and p.a in member_ids and p.b in member_ids
        ]
        groups.append(
            DuplicateGroup(
                canonical_id=canon_id,
                canonical_name=cm.entity_name,
                member_ids=sorted_members,
                pairs=sorted(group_pairs, key=lambda p: (p.a, p.b)),
            )
        )
    return sorted(groups, key=lambda g: g.canonical_id)
