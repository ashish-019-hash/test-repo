"""bind_entity(ctx, lexicons, cfg) -> EntityBinding: five scopes in priority order + decay."""

from __future__ import annotations

import re
from collections import Counter

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.attribute import EntityBinding
from doc_extractor.schemas.document import Document
from doc_extractor.scoring.lexicons import KnownEntity, Lexicons
from doc_extractor.scoring.signals import ScoringContext
from doc_extractor.scoring.syntactic import find_binding_pattern

HEADING_ENTITY_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?(.+?)\s+(?:service\s+)?attributes$", re.IGNORECASE)


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def heading_implied_entity(text: str) -> str | None:
    """'4.3 EVC Service Attributes' -> 'EVC'; '5.1 UNI Attributes' -> 'UNI'."""
    m = HEADING_ENTITY_RE.match(text.strip())
    if not m:
        return None
    candidate = m.group(1).strip()
    return candidate or None


def known_entities_for_document(document: Document, lexicons: Lexicons) -> dict[str, KnownEntity]:
    """`known_entities.yaml` entries plus entities implied by "<X> [Service] Attributes" headings."""
    result: dict[str, KnownEntity] = dict(lexicons.known_entities)
    norm_to_key = {_norm(k): k for k in result}
    texts: list[str] = []
    for block in document.blocks:
        if block.heading_level is not None and block.text:
            texts.append(block.text)
        if block.table is not None and block.table.subject_hint:
            texts.append(block.table.subject_hint)
    for text in texts:
        implied = heading_implied_entity(text)
        if not implied or _norm(implied) in norm_to_key:
            continue
        ke = lexicons.known_entity_lookup(implied)
        if ke is not None:
            norm_to_key[_norm(implied)] = ke.name
            continue
        result[implied] = KnownEntity(name=implied, layer=None, type=None, aliases=())
        norm_to_key[_norm(implied)] = implied
    return result


def _norm_map(known_entities: dict[str, KnownEntity]) -> dict[str, str]:
    norm_map: dict[str, str] = {}
    for name, ke in known_entities.items():
        norm_map.setdefault(_norm(name), name)
        for alias in ke.aliases:
            norm_map.setdefault(_norm(alias), name)
    return norm_map


def _resolve_entity_name(text: str, known_entities: dict[str, KnownEntity], lexicons: Lexicons) -> str | None:
    norm_map = _norm_map(known_entities)

    implied = heading_implied_entity(text)
    if implied:
        return norm_map.get(_norm(implied), implied)

    ke = lexicons.known_entity_lookup(text)
    if ke is not None and ke.name in known_entities:
        return ke.name

    for name in sorted(known_entities, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            return name
    return None


def _confidence(cfg: AppConfig, scope: str, distance: int) -> float:
    multiplier = cfg.binding.scope_multipliers[scope]
    decay = cfg.binding.distance_decay
    floor = cfg.binding.floor
    return round(multiplier * max(floor, 1 - decay * distance), 6)


def table_subject_entities(
    document: Document, known_entities: dict[str, KnownEntity], lexicons: Lexicons
) -> set[str]:
    """Normalized entity names declared as the subject of some spec table in `document`."""
    out: set[str] = set()
    for block in document.blocks:
        if block.table is not None and block.table.subject_hint:
            entity = _resolve_entity_name(block.table.subject_hint, known_entities, lexicons)
            if entity:
                out.add(_norm(entity))
    return out


def bind_entity(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> EntityBinding:
    # 1. table_subject
    if ctx.table is not None and ctx.table.subject_hint:
        entity = _resolve_entity_name(ctx.table.subject_hint, ctx.known_entities, lexicons)
        if entity:
            return EntityBinding(
                entity_name=entity,
                scope="table_subject",
                structural_distance=0,
                confidence=_confidence(cfg, "table_subject", 0),
                evidence=ctx.table.subject_hint,
            )

    # 2. nearest_heading (only headings that explicitly name an "<X> [Service] Attributes"
    # section bind here; plain topic headings like "6 Subscriber" are too weak a signal and
    # are left to the syntactic/compound_modifier scopes below)
    norm_map = _norm_map(ctx.known_entities)
    for idx, heading in enumerate(reversed(ctx.heading_path)):
        implied = heading_implied_entity(heading)
        if implied:
            entity = norm_map.get(_norm(implied), implied)
            return EntityBinding(
                entity_name=entity,
                scope="nearest_heading",
                structural_distance=idx,
                confidence=_confidence(cfg, "nearest_heading", idx),
                evidence=heading,
            )

    # 3. syntactic (genitive / copular / RFC-2119 pattern captured in the sentence)
    sentence = ctx.sentence or ctx.candidate.source_text
    found = find_binding_pattern(sentence or "", ctx.candidate.raw_name, list(ctx.known_entities))
    if found:
        entity_name, evidence, _kind = found
        return EntityBinding(
            entity_name=entity_name,
            scope="syntactic",
            structural_distance=0,
            confidence=_confidence(cfg, "syntactic", 0),
            evidence=evidence,
        )

    # 4. compound_modifier ("EVC ID" -> "EVC" + property-noun "ID")
    name = ctx.candidate.raw_name
    for entity_name in sorted(ctx.known_entities, key=len, reverse=True):
        prefix = entity_name + " "
        if name.lower().startswith(prefix.lower()) and len(name) > len(prefix):
            remainder = name[len(prefix) :].strip()
            tokens = remainder.split()
            head = tokens[-1].lower() if tokens else ""
            if lexicons.is_property_noun(head) or lexicons.gazetteer_match(remainder):
                return EntityBinding(
                    entity_name=entity_name,
                    scope="compound_modifier",
                    structural_distance=0,
                    confidence=_confidence(cfg, "compound_modifier", 0),
                    evidence=name,
                )

    # 5. document_default
    default_entity = cfg.binding.document_default_entity
    if not default_entity and ctx.document_title:
        counts: Counter[str] = Counter()
        for entity_name in ctx.known_entities:
            n = len(re.findall(rf"\b{re.escape(entity_name)}\b", ctx.document_title, re.IGNORECASE))
            if n:
                counts[entity_name] = n
        if counts:
            default_entity = counts.most_common(1)[0][0]
    if default_entity:
        return EntityBinding(
            entity_name=default_entity,
            scope="document_default",
            structural_distance=0,
            confidence=_confidence(cfg, "document_default", 0),
            evidence=ctx.document_title,
        )

    # dangling
    return EntityBinding(
        entity_name=None, scope="dangling", structural_distance=0, confidence=0.0, evidence=None
    )


__all__ = [
    "bind_entity",
    "heading_implied_entity",
    "known_entities_for_document",
    "table_subject_entities",
]
