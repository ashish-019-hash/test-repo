"""Shared scoring context and helpers used by every signal function.

Each signal lives in its own module (`structural.py`, `lexical.py`, `value_domain.py`,
`syntactic.py`, `negative.py`) as a pure function ``(ctx, lexicons, cfg) -> SignalHit | None``.
This module only holds the shared `ScoringContext` dataclass, the registry that
`engine.py` iterates over, and small text helpers every signal needs (sentence
splitting, head-noun extraction).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.attribute import AttributeCandidate, SignalHit
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.document import Block, Table
from doc_extractor.scoring.lexicons import KnownEntity, Lexicons

_PAREN_RE = re.compile(r"\([^)]*\)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence boundaries as `(start, end)` offsets into `text` (no whitespace trimmed).

    Because the separator regex itself is consumed by `finditer`, `text[start:end]` for
    consecutive spans plus the exact original separator reproduces `text` byte for byte,
    so slicing a *range* of spans is always a literal substring of `text`.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def merged_span_for(text: str, needle: str) -> str | None:
    """Slice `text` from the first to the last sentence containing `needle` (word-ish, case-insensitive).

    Returns `None` if `needle` does not occur. The result is always a literal substring
    of `text`, suitable as `Evidence.text` / `AttributeCandidate.source_text`.
    """
    pattern = re.compile(rf"\b{re.escape(needle)}s?\b", re.IGNORECASE)
    spans = sentence_spans(text)
    matching = [i for i, (s, e) in enumerate(spans) if pattern.search(text[s:e])]
    if not matching:
        return None
    first, last = spans[matching[0]][0], spans[matching[-1]][1]
    return text[first:last].strip()


def head_noun(name: str) -> str:
    """Last token of `name` after stripping parenthetical asides, lowercased."""
    stripped = _PAREN_RE.sub("", name).strip()
    tokens = re.findall(r"[A-Za-z0-9']+", stripped)
    return tokens[-1].lower() if tokens else stripped.lower()


def word_tokens(name: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", name)


@dataclass
class ScoringContext:
    """Everything a signal function needs to judge one `AttributeCandidate`."""

    candidate: AttributeCandidate
    chunk: Chunk
    block: Block | None
    heading_path: list[str]
    table: Table | None
    sentence: str | None
    document_title: str | None
    all_candidates: list[AttributeCandidate] = field(default_factory=list)
    known_entities: dict[str, KnownEntity] = field(default_factory=dict)
    # Normalized (casefolded) entity names that are the declared subject of some
    # spec table in the document (from Table.subject_hint / heading-implied entity).
    table_subjects: set[str] = field(default_factory=set)
    # Normalized entity name -> count of *other* candidates whose preliminary
    # binding resolves to it. Populated by the agent's binding pre-pass.
    bind_counts: dict[str, int] = field(default_factory=dict)


SignalFn = Callable[[ScoringContext, Lexicons, AppConfig], SignalHit | None]


def hit(cfg: AppConfig, key: str, evidence: str) -> SignalHit:
    return SignalHit(name=key, weight=cfg.scoring.weights[key], evidence=evidence)


def build_registry() -> list[SignalFn]:
    """Import signal modules lazily to avoid a circular import at module load time."""
    from doc_extractor.scoring import lexical, negative, structural, syntactic, value_domain

    return [
        structural.spec_table_row,
        structural.key_value_pair,
        structural.attribute_heading,
        lexical.property_noun_head,
        lexical.gazetteer_hit,
        value_domain.typed_value,
        value_domain.enum_range_default,
        value_domain.cardinality_marker,
        syntactic.binding_pattern,
        negative.has_sub_attributes,
        negative.lifecycle_subject,
        negative.counted_instances,
        negative.verb_nominalisation,
        negative.furniture,
    ]


__all__ = [
    "ScoringContext",
    "SignalFn",
    "build_registry",
    "head_noun",
    "hit",
    "merged_span_for",
    "sentence_spans",
    "split_sentences",
    "word_tokens",
]
