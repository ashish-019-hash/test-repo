"""Syntactic patterns that bind a candidate name to a known entity in prose.

`entity_alternation` / `PROSE_PATTERNS` are shared with `candidates.py` (candidate
generation) and `binding.py` (scope 3 resolution) so the same grammar is used to
*propose* a name and to *bind* it.
"""

from __future__ import annotations

import re

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.attribute import SignalHit
from doc_extractor.scoring.lexicons import Lexicons
from doc_extractor.scoring.signals import ScoringContext, hit

_X_TOKEN = r"[A-Za-z][A-Za-z\-]*"
_X_STOP = r"(?=\s+(?:is|are|was|were|takes|of|and|in|such)\b|[.,;]|$)"
_X_GROUP = rf"(?P<x>{_X_TOKEN}(?:\s+{_X_TOKEN})?){_X_STOP}"


def entity_alternation(entity_names: list[str]) -> str:
    ordered = sorted(set(entity_names), key=len, reverse=True)
    return "|".join(re.escape(e) for e in ordered)


def compile_patterns(entity_names: list[str]) -> dict[str, re.Pattern[str]]:
    e_alt = entity_alternation(entity_names)
    if not e_alt:
        e_alt = r"(?!x)x"  # matches nothing
    return {
        "rfc2119": re.compile(
            rf"\b(?:each\s+)?(?P<entity>{e_alt})\s+(?:SHALL|MUST)\s+have\s+(?P<tail>[^.!?]*)",
            re.IGNORECASE,
        ),
        "genitive": re.compile(rf"\b(?P<entity>{e_alt})'s\s+{_X_GROUP}", re.IGNORECASE),
        "genitive_of": re.compile(
            rf"\b{_X_GROUP.replace('?P<x>', '?P<x2>')}\s+of\s+the\s+(?P<entity>{e_alt})\b", re.IGNORECASE
        ),
        "copular": re.compile(rf"\bthe\s+(?P<entity>{e_alt})\s+has\s+(?:an?\s+)?{_X_GROUP}", re.IGNORECASE),
    }


def split_rfc2119_tail(tail: str) -> list[str]:
    """'an IMSI and an MSISDN' -> ['IMSI', 'MSISDN']."""
    parts = re.split(r"\band\b", tail)
    out = []
    for p in parts:
        p = re.sub(r"^\s*(?:an?|the)\s+", "", p.strip(), flags=re.IGNORECASE).strip(" .,;")
        if p:
            out.append(p)
    return out


def find_binding_pattern(sentence: str, name: str, entity_names: list[str]) -> tuple[str, str, str] | None:
    """Does `name` bind to one of `entity_names` via a syntactic pattern in `sentence`?

    Returns `(entity_name, evidence_text, pattern_kind)` or `None`.
    """
    if not sentence:
        return None
    name_re = re.escape(name)
    patterns = compile_patterns(entity_names)

    for m in patterns["genitive"].finditer(sentence):
        if m.group("x").lower() == name.lower():
            return m.group("entity"), m.group(0), "genitive"
    for m in patterns["genitive_of"].finditer(sentence):
        if m.group("x2").lower() == name.lower():
            return m.group("entity"), m.group(0), "genitive"
    for m in patterns["copular"].finditer(sentence):
        if m.group("x").lower() == name.lower():
            return m.group("entity"), m.group(0), "copular"
    for m in patterns["rfc2119"].finditer(sentence):
        for item in split_rfc2119_tail(m.group("tail")):
            if item.lower() == name.lower() or re.fullmatch(rf"{name_re}s?", item, re.IGNORECASE):
                return m.group("entity"), m.group(0), "rfc2119"
    return None


def binding_pattern(ctx: ScoringContext, lexicons: Lexicons, cfg: AppConfig) -> SignalHit | None:
    if ctx.candidate.origin != "prose_regex":
        return None
    sentence = ctx.sentence or ctx.candidate.source_text
    if not sentence:
        return None
    found = find_binding_pattern(sentence, ctx.candidate.raw_name, list(ctx.known_entities))
    if found is None:
        return None
    entity_name, evidence, kind = found
    return hit(cfg, "syntactic.binding_pattern", f'{kind} pattern binds to "{entity_name}": "{evidence}"')


__all__ = [
    "binding_pattern",
    "compile_patterns",
    "entity_alternation",
    "find_binding_pattern",
    "split_rfc2119_tail",
]
