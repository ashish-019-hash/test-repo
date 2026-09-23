"""PDF ingestion: pdfplumber for text/tables (word-level layout), pypdf for metadata."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import pdfplumber
import pypdf

from doc_extractor.config.models import AppConfig
from doc_extractor.ingest.base import render_table_text, resolve_table_captions
from doc_extractor.ingest.cleaning import (
    collapse_whitespace,
    dehyphenate,
    load_furniture_patterns,
    mark_furniture,
)
from doc_extractor.ingest.ocr import ocr_available, ocr_page
from doc_extractor.schemas.document import Block, Document, Page, Table
from doc_extractor.storage import ids

_HEADING_RATIO = 1.15
_GAP_MULTIPLIER = 1.6


def _flush_pending_lines(
    pending: list[dict[str, Any]],
    page_index: int,
    rank: dict[float, int],
    builder: _BlockBuilder,
    blocks: list[Block],
) -> None:
    """Merge buffered same-font-size lines into one heading/paragraph block."""
    if not pending:
        return
    size = pending[0]["size"]
    text = collapse_whitespace(dehyphenate("\n".join(ln["text"] for ln in pending)))
    level = rank.get(size)
    kind = "heading" if level else "paragraph"
    bbox_ = (
        min(ln["x0"] for ln in pending),
        min(ln["top"] for ln in pending),
        max(ln["x1"] for ln in pending),
        max(ln["bottom"] for ln in pending),
    )
    blocks.append(builder.new(page_index, kind, text, level, bbox_))
    pending.clear()


def _extract_metadata(path: Path) -> tuple[dict[str, str], int]:
    reader = pypdf.PdfReader(str(path))
    meta = reader.metadata
    out: dict[str, str] = {}
    if meta is not None:
        if meta.title:
            out["title"] = str(meta.title)
        if meta.author:
            out["author"] = str(meta.author)
        if meta.creator:
            out["creator"] = str(meta.creator)
    return out, len(reader.pages)


def _group_words_into_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_top: float | None = None
    for w in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        top = round(w["top"], 1)
        if current and current_top is not None and abs(top - current_top) > 2.0:
            lines.append(current)
            current = []
        current.append(w)
        current_top = top
    if current:
        lines.append(current)
    return [_finalize_line(ln) for ln in lines]


def _finalize_line(words: list[dict[str, Any]]) -> dict[str, Any]:
    words_sorted = sorted(words, key=lambda w: w["x0"])
    text = " ".join(w["text"] for w in words_sorted)
    sizes = [round(float(w["size"]), 1) for w in words_sorted]
    size = statistics.mode(sizes)
    return {
        "text": text,
        "size": size,
        "x0": min(w["x0"] for w in words_sorted),
        "x1": max(w["x1"] for w in words_sorted),
        "top": min(w["top"] for w in words_sorted),
        "bottom": max(w["bottom"] for w in words_sorted),
    }


def _within_any_table(line: dict[str, Any], table_bboxes: list[tuple[float, float, float, float]]) -> bool:
    center = (line["top"] + line["bottom"]) / 2.0
    return any(bbox[1] - 1.0 <= center <= bbox[3] + 1.0 for bbox in table_bboxes)


def _clean_cell(text: str | None) -> str:
    return " ".join((text or "").split())


class _BlockBuilder:
    """Accumulates per-page block counters and produces `Block`s with deterministic ids."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self._counters: dict[int, int] = {}

    def new(
        self,
        page: int,
        kind: str,
        text: str,
        heading_level: int | None,
        bbox: tuple[float, float, float, float] | None,
        table: Table | None = None,
    ) -> Block:
        n = self._counters.get(page, 0)
        self._counters[page] = n + 1
        return Block(
            block_id=ids.block_id(self.document_id, page, n),
            page=page,
            kind=kind,  # type: ignore[arg-type]
            text=text,
            heading_level=heading_level,
            bbox=bbox,
            table=table,
        )


def _load(path: Path, cfg: AppConfig, document_id: str) -> Document:
    metadata, _page_count = _extract_metadata(path)

    with pdfplumber.open(path) as pdf:
        per_page: list[tuple[list[dict[str, Any]], list[Any], list[tuple[float, float, float, float]]]] = []
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size"])
            tables = page.find_tables()
            table_bboxes: list[tuple[float, float, float, float]] = [
                (float(t.bbox[0]), float(t.bbox[1]), float(t.bbox[2]), float(t.bbox[3])) for t in tables
            ]
            lines = _group_words_into_lines(words)
            body_lines = [ln for ln in lines if not _within_any_table(ln, table_bboxes)]
            per_page.append((body_lines, tables, table_bboxes))

        # Body-size mode is computed from non-table text only: table cell font sizes are
        # usually smaller/more numerous and would otherwise skew the paragraph baseline.
        all_sizes = [ln["size"] for body_lines, _, _ in per_page for ln in body_lines]
        body_size = statistics.mode(all_sizes) if all_sizes else 10.0
        threshold = body_size * _HEADING_RATIO
        heading_sizes = sorted({s for s in all_sizes if s >= threshold}, reverse=True)
        rank = {s: i + 1 for i, s in enumerate(heading_sizes)}

        builder = _BlockBuilder(document_id)
        blocks: list[Block] = []

        for page_index in range(1, len(pdf.pages) + 1):
            body_lines, tables, table_bboxes = per_page[page_index - 1]

            entries: list[tuple[float, str, Any]] = [(ln["top"], "line", ln) for ln in body_lines]
            entries.extend(
                (bbox[1], "table", (t, bbox)) for t, bbox in zip(tables, table_bboxes, strict=True)
            )
            entries.sort(key=lambda e: e[0])

            pending: list[dict[str, Any]] = []

            for _, kind, payload in entries:
                if kind == "table":
                    _flush_pending_lines(pending, page_index, rank, builder, blocks)
                    table_obj, bbox = payload
                    rows = [[_clean_cell(c) for c in row] for row in (table_obj.extract() or [])]
                    header, *data_rows = rows if rows else [[]]
                    table_model = Table(header=header, rows=data_rows)
                    text_ = render_table_text(header, data_rows)
                    blocks.append(builder.new(page_index, "table", text_, None, bbox, table=table_model))
                    continue
                line = payload
                if pending:
                    same_size = abs(line["size"] - pending[0]["size"]) <= 0.5
                    gap = line["top"] - pending[-1]["top"]
                    if not same_size or gap > pending[0]["size"] * _GAP_MULTIPLIER:
                        _flush_pending_lines(pending, page_index, rank, builder, blocks)
                pending.append(line)
            _flush_pending_lines(pending, page_index, rank, builder, blocks)

        blocks = resolve_table_captions(blocks)
        blocks = mark_furniture(blocks, load_furniture_patterns(cfg))

        pages: list[Page] = []
        for page_index in range(1, len(pdf.pages) + 1):
            page_blocks = [b for b in blocks if b.page == page_index]
            text = "\n".join(b.text for b in page_blocks)
            ocr_required = len(text.strip()) < cfg.ingest.min_chars_per_page
            if ocr_required and cfg.ingest.ocr_enabled and ocr_available():
                ocr_text = ocr_page(path, page_index)
                if ocr_text.strip():
                    text = ocr_text
                    ocr_required = len(text.strip()) < cfg.ingest.min_chars_per_page
            pages.append(
                Page(
                    page_number=page_index,
                    text=text,
                    ocr_required=ocr_required,
                    block_ids=[b.block_id for b in page_blocks],
                )
            )

    ocr_required_count = sum(1 for p in pages if p.ocr_required)
    ratio = ocr_required_count / len(pages) if pages else 0.0
    needs_ocr = ratio > cfg.ingest.ocr_page_ratio
    text_coverage = 1.0 - ratio

    title = metadata.get("title") or next((b.text for b in blocks if b.heading_level == 1), None)

    return Document(
        document_id=document_id,
        file_name=path.name,
        format="pdf",
        title=title,
        metadata=metadata,
        pages=pages,
        blocks=blocks,
        needs_ocr=needs_ocr,
        text_coverage=text_coverage,
    )


class _PdfLoader:
    def load(self, path: Path, cfg: AppConfig, document_id: str) -> Document:
        return _load(path, cfg, document_id)


LOADER = _PdfLoader()
