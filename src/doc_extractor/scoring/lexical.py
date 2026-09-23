"""Lexical signals: property-noun heads and gazetteer membership."""

from __future__ import annotations

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.attribute import SignalHit
from doc_extractor.scoring.lexicons import Lexicons
from doc_extractor.scoring.signals import ScoringContext, head_noun, hit


def property_noun_head(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    name = ctx.candidate.raw_name
    candidates_for_head = [name]
    canonical = lexicons.gazetteer_match(name)
    if canonical:
        candidates_for_head.extend(lexicons.gazetteer.get(canonical, []))
    alias_canonical = lexicons.alias_canonical(name)
    if alias_canonical:
        candidates_for_head.append(alias_canonical)
        candidates_for_head.extend(lexicons.aliases.get(alias_canonical, []))
    seen: set[str] = set()
    for text in candidates_for_head:
        h = head_noun(text)
        if h in seen:
            continue
        seen.add(h)
        if lexicons.is_property_noun(h):
            return hit(
                cfg, "lexical.property_noun_head", f'head noun "{h}" (from "{text}") is a property noun'
            )
    return None


def gazetteer_hit(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    canonical = lexicons.gazetteer_match(ctx.candidate.raw_name)
    if canonical is None:
        return None
    return hit(
        cfg, "lexical.gazetteer_hit", f'"{ctx.candidate.raw_name}" matches gazetteer entry "{canonical}"'
    )


__all__ = ["gazetteer_hit", "property_noun_head"]
