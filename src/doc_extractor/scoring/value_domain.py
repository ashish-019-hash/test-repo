"""Value-domain signals: typed values, enum/range/default markers, cardinality markers."""

from __future__ import annotations

import re

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.attribute import SignalHit
from doc_extractor.scoring.lexicons import Lexicons
from doc_extractor.scoring.signals import ScoringContext, hit

_NUMBER_UNIT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:Mbps|Gbps|kbps|bps|ms|bytes|KB|MB|GB|%)\b", re.IGNORECASE)
_E164_RE = re.compile(r"\+\d{7,15}\b")
_CIDR_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b")
_MAC_RE = re.compile(r"\b[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5}\b")
_BOOLEAN_RE = re.compile(r"\b(?:true|false)\b", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

_TYPED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_E164_RE, "E.164"),
    (_CIDR_RE, "CIDR"),
    (_MAC_RE, "MAC"),
    (_BOOLEAN_RE, "boolean"),
    (_ISO_DATE_RE, "ISO-8601"),
    (_NUMBER_UNIT_RE, "number+unit"),
]

_RANGE_RE = re.compile(r"\d+\s*[\u2013\u2014-]\s*\d+")
_ONE_OF_RE = re.compile(r"\bone of\b", re.IGNORECASE)
_DEFAULT_WORD_RE = re.compile(r"\bdefault(?:s|ed|ing)?\b", re.IGNORECASE)

_CARDINALITY_BRACKET_RE = re.compile(r"\[\s*(?:0\.\.1|1\.\.\*|\d+\.\.(?:\d+|\*))\s*\]")
_MOC_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])[MOC](?![A-Za-z0-9])")


def _text_of(ctx: ScoringContext) -> str:
    """Text to regex-scan when the relevant cell is absent.

    Spec-table rows fall back to their *own* cell values only (never the whole
    rendered table / chunk text), so a range/default/unit in a sibling row never
    leaks into this row's signals.
    """
    if ctx.sentence:
        return ctx.sentence
    if ctx.candidate.origin == "spec_table":
        return " ".join(ctx.candidate.cells.values())
    return ctx.candidate.source_text or ""


def typed_value(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    type_cell = ctx.candidate.cells.get("type")
    if type_cell:
        return hit(cfg, "value_domain.typed_value", f'type cell = "{type_cell}"')
    text = _text_of(ctx)
    for pattern, label in _TYPED_PATTERNS:
        m = pattern.search(text)
        if m:
            return hit(cfg, "value_domain.typed_value", f'{label} pattern matched "{m.group(0)}"')
    return None


def enum_range_default(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    cells = ctx.candidate.cells
    if cells.get("range"):
        return hit(cfg, "value_domain.enum_range_default", f'range cell = "{cells["range"]}"')
    if cells.get("default"):
        return hit(cfg, "value_domain.enum_range_default", f'default cell = "{cells["default"]}"')
    text = _text_of(ctx)
    m = _RANGE_RE.search(text)
    if m:
        return hit(cfg, "value_domain.enum_range_default", f'range pattern "{m.group(0)}"')
    if _ONE_OF_RE.search(text):
        return hit(cfg, "value_domain.enum_range_default", '"one of" in text')
    if _DEFAULT_WORD_RE.search(text):
        return hit(cfg, "value_domain.enum_range_default", '"default" in text')
    return None


def cardinality_marker(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    opt = (ctx.candidate.cells.get("optionality") or "").strip().upper()
    if opt in {"M", "O", "C"}:
        return hit(cfg, "value_domain.cardinality_marker", f'optionality cell = "{opt}"')
    text = _text_of(ctx)
    m = _CARDINALITY_BRACKET_RE.search(text)
    if m:
        return hit(cfg, "value_domain.cardinality_marker", f'cardinality bracket "{m.group(0)}"')
    m = _MOC_LETTER_RE.search(text)
    if m:
        return hit(cfg, "value_domain.cardinality_marker", f'standalone marker "{m.group(0)}"')
    return None


__all__ = ["cardinality_marker", "enum_range_default", "typed_value"]
