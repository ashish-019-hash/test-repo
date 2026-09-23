"""Unit tests for the text/markdown ingest loader."""

from __future__ import annotations

from pathlib import Path

from doc_extractor.config.models import AppConfig
from doc_extractor.ingest import load_document


def test_load_markdown_title_and_headings(cfg: AppConfig, telecom_spec_md: Path) -> None:
    doc = load_document(telecom_spec_md, cfg)
    assert doc.format == "md"
    assert doc.title == "Carrier Ethernet Service Specification v2.1"
    headings = [b for b in doc.blocks if b.kind == "heading"]
    assert any(h.text == "4.3 EVC Service Attributes" and h.heading_level == 3 for h in headings)
    assert any(h.text == "5.1 UNI Attributes" and h.heading_level == 3 for h in headings)


def test_load_markdown_document_id_is_stable(cfg: AppConfig, telecom_spec_md: Path) -> None:
    doc1 = load_document(telecom_spec_md, cfg)
    doc2 = load_document(telecom_spec_md, cfg)
    assert doc1.document_id == doc2.document_id
    assert doc1.document_id.startswith("doc-")


def test_load_markdown_evc_table(cfg: AppConfig, telecom_spec_md: Path) -> None:
    doc = load_document(telecom_spec_md, cfg)
    tables = [b for b in doc.blocks if b.kind == "table"]
    evc_table = next(t for t in tables if t.table is not None and t.table.rows[0][0] == "CIR")
    assert evc_table.table is not None
    assert evc_table.table.header == [
        "Attribute",
        "Type",
        "Units",
        "M/O/C",
        "Range",
        "Default",
        "Description",
    ]
    assert "| CIR | Integer" in evc_table.text
    assert evc_table.table.subject_hint is not None
    assert "EVC" in evc_table.table.subject_hint


def test_load_markdown_furniture_marked(cfg: AppConfig, telecom_spec_md: Path) -> None:
    doc = load_document(telecom_spec_md, cfg)
    furniture_texts = {b.text for b in doc.blocks if b.kind == "furniture"}
    assert "Revision History" in furniture_texts
    assert "Page 3 of 3" in furniture_texts


def test_load_markdown_blocks_never_deleted(cfg: AppConfig, telecom_spec_md: Path) -> None:
    doc = load_document(telecom_spec_md, cfg)
    all_texts = [b.text for b in doc.blocks]
    assert "Revision History" in all_texts
