"""score(ctx, lexicons, cfg) -> ScoreBreakdown: sum weights, clamp, short-circuit, route."""

from __future__ import annotations

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.attribute import ScoreBreakdown
from doc_extractor.scoring.lexicons import Lexicons
from doc_extractor.scoring.signals import ScoringContext, build_registry

_POPULATED_CELL_KEYS = ("type", "units", "optionality", "range", "default")


def score(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> ScoreBreakdown:
    hits = [h for h in (fn(ctx, lexicons, cfg) for fn in build_registry()) if h is not None]
    raw_sum = round(sum(h.weight for h in hits), 10)

    clamp_cfg = cfg.scoring.clamp
    clamped = min(clamp_cfg.max, max(clamp_cfg.min, raw_sum))

    short_circuit = False
    sc_cfg = cfg.scoring.short_circuit
    if sc_cfg.enabled:
        has_spec_row = any(h.name == "structural.spec_table_row" for h in hits)
        has_negative = any(h.weight < 0 for h in hits)
        populated = sum(1 for k in _POPULATED_CELL_KEYS if ctx.candidate.cells.get(k))
        name_tokens = len(ctx.candidate.raw_name.split())
        if (
            has_spec_row
            and not has_negative
            and populated >= sc_cfg.min_populated_cells
            and name_tokens <= sc_cfg.max_name_tokens
        ):
            clamped = max(clamped, sc_cfg.confidence)
            short_circuit = True

    routing = cfg.scoring.routing
    if clamped >= routing.accept:
        route: str = "accept"
    elif clamped >= routing.review:
        route = "review"
    else:
        route = "discard"

    discard_reason = None
    if route == "discard":
        promoted = any(h.name == "negative.has_sub_attributes" for h in hits)
        discard_reason = "promoted" if promoted else "below_threshold"

    return ScoreBreakdown(
        hits=hits,
        raw_sum=raw_sum,
        clamped=clamped,
        short_circuit=short_circuit,
        route=route,  # type: ignore[arg-type]
        discard_reason=discard_reason,
    )


__all__ = ["score"]
