"""Unit tests for the PDF ingest loader."""

from __future__ import annotations

from pathlib import Path

from doc_extractor.config.models import AppConfig
from doc_extractor.ingest import load_document


def test_load_pdf_title_and_headings(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "telecom_spec.pdf", cfg)
    assert doc.format == "pdf"
    assert doc.title == "Carrier Ethernet Service Specification v2.1"
    headings = [b for b in doc.blocks if b.kind == "heading"]
    assert any(h.text == "4.3 EVC Service Attributes" and h.heading_level == 3 for h in headings)
    assert any(h.text == "5.1 UNI Attributes" and h.heading_level == 3 for h in headings)


def test_load_pdf_evc_table(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "telecom_spec.pdf", cfg)
    tables = [b for b in doc.blocks if b.kind == "table"]
    evc_table = next(t for t in tables if t.table is not None and t.table.rows[0][0] == "CIR")
    assert evc_table.table is not None
    assert evc_table.table.header[0] == "Attribute"
    assert evc_table.table.subject_hint is not None
    assert "EVC" in evc_table.table.subject_hint
    assert "| CIR | Integer" in evc_table.text


def test_load_pdf_footer_marked_furniture(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "telecom_spec.pdf", cfg)
    furniture_texts = {b.text for b in doc.blocks if b.kind == "furniture"}
    assert "Page 1 of 2" in furniture_texts
    assert "Page 2 of 2" in furniture_texts


def test_load_pdf_no_duplicate_table_text(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "telecom_spec.pdf", cfg)
    paragraphs = [b for b in doc.blocks if b.kind == "paragraph"]
    assert not any("Committed Information Rate" in p.text for p in paragraphs)


def test_load_pdf_two_pages(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "telecom_spec.pdf", cfg)
    assert len(doc.pages) == 2
    assert not doc.needs_ocr
    assert doc.text_coverage == 1.0


def test_load_blank_pdf_flags_ocr(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "blank_page.pdf", cfg)
    assert doc.format == "pdf"
    assert len(doc.pages) == 1
    assert doc.pages[0].ocr_required
    assert doc.needs_ocr
