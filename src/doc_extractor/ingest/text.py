"""TXT/MD ingestion: `#` headings, pipe tables, blank-line paragraphs, simple list items."""

from __future__ import annotations

import re
from pathlib import Path

from doc_extractor.config.models import AppConfig
from doc_extractor.ingest.base import render_table_text, resolve_table_captions
from doc_extractor.ingest.cleaning import collapse_whitespace, load_furniture_patterns, mark_furniture
from doc_extractor.schemas.document import Block, Document, Page, Table
from doc_extractor.storage import ids

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
_SEPARATOR_CELL_RE = re.compile(r"^-{2,}$")


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _load(path: Path, cfg: AppConfig, document_id: str) -> Document:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    blocks: list[Block] = []
    counter = 0

    def new_block(kind: str, text: str, heading_level: int | None = None, table: Table | None = None) -> None:
        nonlocal counter
        blocks.append(
            Block(
                block_id=ids.block_id(document_id, 1, counter),
                page=1,
                kind=kind,  # type: ignore[arg-type]
                text=text,
                heading_level=heading_level,
                table=table,
            )
        )
        counter += 1

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        stripped = line.lstrip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            new_block("heading", stripped[level:].strip(), heading_level=level)
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [_split_row(tl) for tl in table_lines]
            rows = [r for r in rows if not all(_SEPARATOR_CELL_RE.fullmatch(c) for c in r)]
            if rows:
                header, *data_rows = rows
                new_block(
                    "table", render_table_text(header, data_rows), table=Table(header=header, rows=data_rows)
                )
            continue
        if _LIST_ITEM_RE.match(stripped):
            while i < n and lines[i].strip() and _LIST_ITEM_RE.match(lines[i].lstrip()):
                item_text = _LIST_ITEM_RE.sub("", lines[i].strip(), count=1).strip()
                new_block("list_item", item_text)
                i += 1
            continue
        para_lines = [line.strip()]
        i += 1
        while (
            i < n
            and lines[i].strip()
            and not lines[i].lstrip().startswith(("#", "|"))
            and not _LIST_ITEM_RE.match(lines[i].lstrip())
        ):
            para_lines.append(lines[i].strip())
            i += 1
        new_block("paragraph", collapse_whitespace(" ".join(para_lines)))

    blocks = resolve_table_captions(blocks)
    blocks = mark_furniture(blocks, load_furniture_patterns(cfg))

    text_all = "\n".join(b.text for b in blocks)
    ocr_required = len(text_all.strip()) < cfg.ingest.min_chars_per_page
    page = Page(
        page_number=1, text=text_all, ocr_required=ocr_required, block_ids=[b.block_id for b in blocks]
    )

    fmt = "md" if path.suffix.lower() in (".md", ".markdown") else "txt"
    title = next((b.text for b in blocks if b.heading_level == 1), None)

    return Document(
        document_id=document_id,
        file_name=path.name,
        format=fmt,  # type: ignore[arg-type]
        title=title,
        metadata={},
        pages=[page],
        blocks=blocks,
        needs_ocr=ocr_required,
        text_coverage=0.0 if ocr_required else 1.0,
    )


class _TextLoader:
    def load(self, path: Path, cfg: AppConfig, document_id: str) -> Document:
        return _load(path, cfg, document_id)


LOADER = _TextLoader()
