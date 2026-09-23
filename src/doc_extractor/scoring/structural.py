"""Structural signals: table/heading shape, independent of word meaning."""

from __future__ import annotations

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.attribute import SignalHit
from doc_extractor.scoring.lexicons import Lexicons
from doc_extractor.scoring.signals import ScoringContext, hit

_ROW_HEADER_CATEGORIES = ("name", "type", "units", "optionality", "default", "description")


def spec_table_row(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    if ctx.candidate.origin != "spec_table":
        return None
    matched = sum(1 for k in _ROW_HEADER_CATEGORIES if k in ctx.candidate.cells)
    if matched < 2:
        return None
    return hit(cfg, "structural.spec_table_row", f"spec table header matched {matched} categories")


def key_value_pair(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    if ctx.candidate.origin not in ("kv_pair", "form_field"):
        return None
    return hit(cfg, "structural.key_value_pair", f"origin={ctx.candidate.origin}")


def attribute_heading(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    for heading in ctx.heading_path:
        if lexicons.attribute_heading_regex.search(heading):
            return hit(cfg, "structural.attribute_heading", f'heading "{heading}" matches')
    return None


__all__ = ["attribute_heading", "key_value_pair", "spec_table_row"]
