"""Negative signals: evidence that a candidate is not really an attribute."""

from __future__ import annotations

import re

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.attribute import SignalHit
from doc_extractor.scoring.lexicons import Lexicons
from doc_extractor.scoring.signals import ScoringContext, head_noun, hit

_NOMINALISATION_SUFFIX_RE = re.compile(r"(?:tion|ing|ment|ance)$", re.IGNORECASE)


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def has_sub_attributes(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    name = _norm(ctx.candidate.raw_name)
    count = ctx.bind_counts.get(name, 0)
    if count >= cfg.entity_generation.promotion_min_sub_attributes:
        return hit(
            cfg,
            "negative.has_sub_attributes",
            f'{count} other candidate(s) bind to "{ctx.candidate.raw_name}" as their entity',
        )
    if name in ctx.table_subjects:
        return hit(
            cfg,
            "negative.has_sub_attributes",
            f'"{ctx.candidate.raw_name}" is the subject of another spec table',
        )
    return None


def lifecycle_subject(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    sentence = ctx.sentence or ctx.candidate.source_text or ""
    name_re = re.escape(ctx.candidate.raw_name)
    m = re.search(
        rf"\b(?:the\s+)?{name_re}\s+(?:is|are|was|were)\s+(?P<verbs>[a-z][a-z,\s]*?)(?:\.|$)",
        sentence,
        re.IGNORECASE,
    )
    if not m:
        return None
    verbs = re.findall(r"[a-z]+", m.group("verbs").lower())
    fired = [v for v in verbs if lexicons.is_lifecycle_verb(v)]
    if not fired:
        return None
    return hit(
        cfg,
        "negative.lifecycle_subject",
        f'"{ctx.candidate.raw_name}" is subject of lifecycle verb(s) {fired}',
    )


def counted_instances(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    text = ctx.sentence or ctx.candidate.source_text or ""
    name_re = re.escape(ctx.candidate.raw_name)
    pattern = re.compile(
        rf"\b(?:\d+|multiple|several|number of)\s+(?:[a-z]+\s+)?{name_re}s?\b", re.IGNORECASE
    )
    m = pattern.search(text)
    if not m:
        return None
    return hit(cfg, "negative.counted_instances", f'counted-instance pattern "{m.group(0)}"')


def verb_nominalisation(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    name = ctx.candidate.raw_name
    if not _NOMINALISATION_SUFFIX_RE.search(name):
        return None
    head = head_noun(name)
    if lexicons.is_property_noun(head):
        return None
    return hit(cfg, "negative.verb_nominalisation", f'"{name}" ends in a nominalising suffix')


def furniture(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    if ctx.block is not None and ctx.block.kind == "furniture":
        return hit(cfg, "negative.furniture", f'block "{ctx.block.block_id}" is kind=furniture')
    if lexicons.is_furniture(ctx.candidate.raw_name):
        return hit(cfg, "negative.furniture", f'"{ctx.candidate.raw_name}" matches a furniture pattern')
    if ctx.candidate.source_text and lexicons.is_furniture(ctx.candidate.source_text):
        return hit(cfg, "negative.furniture", f'"{ctx.candidate.source_text}" matches a furniture pattern')
    return None


__all__ = [
    "counted_instances",
    "furniture",
    "has_sub_attributes",
    "lifecycle_subject",
    "verb_nominalisation",
]
