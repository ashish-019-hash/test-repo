"""Unit tests for the DOCX ingest loader."""

from __future__ import annotations

from pathlib import Path

from doc_extractor.config.models import AppConfig
from doc_extractor.ingest import load_document


def test_load_docx_title_and_headings(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "telecom_spec.docx", cfg)
    assert doc.format == "docx"
    assert doc.title == "Carrier Ethernet Service Specification v2.1"
    headings = [b for b in doc.blocks if b.kind == "heading"]
    assert any(h.text == "4.3 EVC Service Attributes" and h.heading_level == 2 for h in headings)
    assert any(h.text == "5.1 UNI Attributes" and h.heading_level == 2 for h in headings)


def test_load_docx_single_page(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "telecom_spec.docx", cfg)
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert all(b.page == 1 for b in doc.blocks)


def test_load_docx_evc_table(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "telecom_spec.docx", cfg)
    tables = [b for b in doc.blocks if b.kind == "table"]
    evc_table = next(t for t in tables if t.table is not None and t.table.rows[0][0] == "CIR")
    assert evc_table.table is not None
    assert evc_table.table.header[0] == "Attribute"
    assert evc_table.table.subject_hint is not None
    assert "EVC" in evc_table.table.subject_hint
    assert "| CIR | Integer" in evc_table.text


def test_load_docx_tables_and_paragraphs_interleaved(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "telecom_spec.docx", cfg)
    kinds = [b.kind for b in doc.blocks]
    # "1 Introduction" heading is followed eventually by a table further down; interleaving
    # means we should see more than one alternation between prose/heading kinds and "table".
    assert kinds.count("table") == 3
