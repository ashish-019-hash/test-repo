"""Loader protocol and helpers shared by the pdf/docx/text ingest modules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.document import Block, Document

TABLE_CAPTION_RE = re.compile(r"^\s*table\s+\d+\b", re.IGNORECASE)
_CAPTION_PREFIX_RE = re.compile(r"^\s*table\s+\d+\s*[-\u2013\u2014:]?\s*", re.IGNORECASE)


@runtime_checkable
class Loader(Protocol):
    """Every format module exposes an object shaped like this."""

    def load(self, path: Path, cfg: AppConfig, document_id: str) -> Document: ...


def is_table_caption(text: str) -> bool:
    return bool(TABLE_CAPTION_RE.match(text.strip()))


def strip_caption_prefix(text: str) -> str:
    """\"Table 3 \u2013 EVC Service Attributes\" -> \"EVC Service Attributes\"."""
    return _CAPTION_PREFIX_RE.sub("", text).strip()


def render_table_text(header: list[str], rows: list[list[str]]) -> str:
    """Pipe-table rendering so a table stays self-contained inside Block.text / Chunk.source_text.

    Evidence substring matching (owned by the scoring group) depends on rows looking like
    literal "| a | b | c |" lines.
    """
    lines = []
    if header:
        lines.append("| " + " | ".join(header) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def resolve_table_subject(
    caption_before: str | None, caption_after: str | None, heading_above: str | None
) -> tuple[str | None, str | None]:
    """(caption, subject_hint) for a table given its immediate neighbour blocks' text.

    Prefers a nearby "Table N ..." caption line (checked before, then after) over the
    nearest heading above the table.
    """
    for candidate in (caption_before, caption_after):
        if candidate and is_table_caption(candidate):
            return candidate, strip_caption_prefix(candidate)
    return None, heading_above


def resolve_table_captions(blocks: list[Block]) -> list[Block]:
    """Second pass over an already-built, ordered block list: fill in each table block's
    `caption`/`subject_hint` from its immediate neighbours or the nearest preceding heading.
    """
    heading_before: list[str | None] = []
    current_heading: str | None = None
    for b in blocks:
        heading_before.append(current_heading)
        if b.kind == "heading":
            current_heading = b.text

    out = list(blocks)
    for i, b in enumerate(blocks):
        if b.kind != "table" or b.table is None:
            continue
        before_text = blocks[i - 1].text if i > 0 and blocks[i - 1].kind != "table" else None
        after_text = blocks[i + 1].text if i + 1 < len(blocks) and blocks[i + 1].kind != "table" else None
        caption, subject_hint = resolve_table_subject(before_text, after_text, heading_before[i])
        new_table = b.table.model_copy(update={"caption": caption, "subject_hint": subject_hint})
        out[i] = b.model_copy(update={"table": new_table})
    return out
