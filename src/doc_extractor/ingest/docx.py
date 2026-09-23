"""DOCX ingestion: iterate the document body in order (paragraphs and tables interleaved)."""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from doc_extractor.config.models import AppConfig
from doc_extractor.ingest.base import render_table_text, resolve_table_captions
from doc_extractor.ingest.cleaning import collapse_whitespace, load_furniture_patterns, mark_furniture
from doc_extractor.schemas.document import Block, Document, Page, Table
from doc_extractor.storage import ids

_LIST_STYLES = ("list bullet", "list number", "list paragraph")


def _heading_level(style_name: str | None) -> int | None:
    name = (style_name or "").strip().lower()
    if name == "title":
        return 0
    if name.startswith("heading"):
        suffix = name[len("heading") :].strip()
        if suffix.isdigit():
            return int(suffix)
    return None


def _is_list_item(style_name: str | None) -> bool:
    return (style_name or "").strip().lower() in _LIST_STYLES


def _load(path: Path, cfg: AppConfig, document_id: str) -> Document:
    doc = DocxDocument(str(path))
    blocks: list[Block] = []
    counter = 0

    def new_block(kind: str, text: str, heading_level: int | None, table: Table | None = None) -> None:
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

    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            para = DocxParagraph(child, doc)
            text = collapse_whitespace(para.text)
            if not text:
                continue
            style_name = para.style.name if para.style is not None else None
            level = _heading_level(style_name)
            if level is not None:
                new_block("heading", text, level)
            elif _is_list_item(style_name):
                new_block("list_item", text, None)
            else:
                new_block("paragraph", text, None)
        elif tag == "tbl":
            table = DocxTable(child, doc)
            rows_all = [[collapse_whitespace(cell.text) for cell in row.cells] for row in table.rows]
            if not rows_all:
                continue
            header, *data_rows = rows_all
            table_model = Table(header=header, rows=data_rows)
            new_block("table", render_table_text(header, data_rows), None, table=table_model)

    blocks = resolve_table_captions(blocks)
    blocks = mark_furniture(blocks, load_furniture_patterns(cfg))

    page_text = "\n".join(b.text for b in blocks)
    ocr_required = len(page_text.strip()) < cfg.ingest.min_chars_per_page
    page = Page(
        page_number=1, text=page_text, ocr_required=ocr_required, block_ids=[b.block_id for b in blocks]
    )

    metadata: dict[str, str] = {}
    core = doc.core_properties
    if core.title:
        metadata["title"] = core.title
    if core.author:
        metadata["author"] = core.author

    title = metadata.get("title") or next((b.text for b in blocks if b.heading_level in (0, 1)), None)

    return Document(
        document_id=document_id,
        file_name=path.name,
        format="docx",
        title=title,
        metadata=metadata,
        pages=[page],
        blocks=blocks,
        needs_ocr=ocr_required,
        text_coverage=0.0 if ocr_required else 1.0,
    )


class _DocxLoader:
    def load(self, path: Path, cfg: AppConfig, document_id: str) -> Document:
        return _load(path, cfg, document_id)


LOADER = _DocxLoader()
