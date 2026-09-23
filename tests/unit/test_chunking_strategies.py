"""Unit tests for chunking strategies (pure `Block[] -> Chunk[]` functions)."""

from __future__ import annotations

from pathlib import Path

from doc_extractor.chunking import chunk_document
from doc_extractor.chunking.strategies import by_page, fixed_tokens, heading_aware
from doc_extractor.chunking.tokens import count_tokens
from doc_extractor.config.models import AppConfig
from doc_extractor.ingest import load_document


def _assert_chunk_invariants(document, chunks) -> None:  # type: ignore[no-untyped-def]
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    assert ids == sorted(ids), "chunk ids must be sorted"
    block_map = document.block_map()
    covered = set()
    for c in chunks:
        assert c.block_ids, "every chunk has >=1 block"
        expected = "\n".join(block_map[bid].text for bid in c.block_ids)
        assert expected == c.source_text
        covered.update(c.block_ids)
    assert covered == {b.block_id for b in document.blocks}


def test_count_tokens_basic() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0


def test_by_page_one_chunk_per_page(cfg: AppConfig, fixtures_dir: Path) -> None:
    doc = load_document(fixtures_dir / "telecom_spec.pdf", cfg)
    chunks = by_page(doc.blocks, doc.document_id, cfg)
    _assert_chunk_invariants(doc, chunks)
    assert len(chunks) == len(doc.pages)
    for c in chunks:
        assert c.strategy == "by_page"


def test_fixed_tokens_packs_and_isolates_tables(cfg: AppConfig, telecom_spec_md: Path) -> None:
    doc = load_document(telecom_spec_md, cfg)
    chunks = fixed_tokens(doc.blocks, doc.document_id, cfg)
    _assert_chunk_invariants(doc, chunks)
    for c in chunks:
        assert c.strategy == "fixed_tokens"
        table_blocks = [bid for bid in c.block_ids if bid.startswith("blk-")]
        table_kinds = {doc.block_map()[bid].kind for bid in table_blocks}
        if "table" in table_kinds:
            assert table_kinds == {"table"}, "a table must never share a chunk with other blocks"


def test_heading_aware_starts_new_chunk_per_heading(cfg: AppConfig, telecom_spec_md: Path) -> None:
    doc = load_document(telecom_spec_md, cfg)
    chunks = heading_aware(doc.blocks, doc.document_id, cfg)
    _assert_chunk_invariants(doc, chunks)

    evc_chunk = next(c for c in chunks if "CIR" in c.source_text)
    assert evc_chunk.heading_path == [
        "Carrier Ethernet Service Specification v2.1",
        "4 Service Attributes",
        "4.3 EVC Service Attributes",
    ]
    for c in chunks:
        assert c.strategy == "heading_aware"
        assert c.heading_path, "heading_aware chunks must carry a heading breadcrumb"


def test_heading_aware_furniture_never_starts_heading_path(cfg: AppConfig, telecom_spec_md: Path) -> None:
    doc = load_document(telecom_spec_md, cfg)
    chunks = heading_aware(doc.blocks, doc.document_id, cfg)
    for c in chunks:
        assert "Revision History" not in c.heading_path


def test_chunk_document_dispatches_on_strategy(cfg: AppConfig, telecom_spec_md: Path) -> None:
    doc = load_document(telecom_spec_md, cfg)
    chunks = chunk_document(doc, cfg)
    assert chunks
    assert all(c.strategy == cfg.chunking.strategy for c in chunks)


def test_chunk_document_by_page_strategy(cfg: AppConfig, telecom_spec_md: Path) -> None:
    doc = load_document(telecom_spec_md, cfg)
    by_page_cfg = cfg.model_copy(update={"chunking": cfg.chunking.model_copy(update={"strategy": "by_page"})})
    chunks = chunk_document(doc, by_page_cfg)
    assert all(c.strategy == "by_page" for c in chunks)
    assert len(chunks) == len(doc.pages)
