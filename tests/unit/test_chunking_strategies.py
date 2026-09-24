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


def _block(doc_id: str, page: int, n: int, text: str, kind: str = "paragraph"):  # type: ignore[no-untyped-def]
    from doc_extractor.schemas.document import Block
    from doc_extractor.storage import ids

    return Block(block_id=ids.block_id(doc_id, page, n), page=page, kind=kind, text=text)  # type: ignore[arg-type]


def _blocks_with_overlap_across_table() -> list:  # type: ignore[type-arg]
    """Reproduces the ICICI credit-card layout: a short trailing paragraph on page 2 is
    carried as overlap, then a table on page 3 forces a flush before more page-3 prose."""
    d = "doc-x"
    return [
        _block(d, 2, 0, "Card variants and fees are listed below."),
        _block(d, 2, 1, "2"),  # page footer -> tiny block, always fits the overlap budget
        _block(d, 3, 0, "| Card | Fee |\n| Coral | 500 |", kind="table"),
        _block(d, 3, 1, "The fees are billed to the card account."),
        _block(d, 3, 2, "3"),
        _block(d, 4, 0, "| Card | Limit |\n| Sapphiro | 1000 |", kind="table"),
        _block(d, 4, 1, "Charges apply irrespective of the variant."),
    ]


def test_overlap_across_table_keeps_chunk_ids_sorted(cfg: AppConfig) -> None:
    """Regression: the chunk after an isolated table opened with carried page-2 overlap and
    received a page-2 id, which sorted before the page-3 table chunk."""
    blocks = _blocks_with_overlap_across_table()
    for fn in (fixed_tokens, heading_aware):
        chunks = fn(blocks, "doc-x", cfg)
        ids = [c.chunk_id for c in chunks]
        assert ids == sorted(ids), ids
        assert len(set(ids)) == len(ids)
        covered = {b for c in chunks for b in c.block_ids}
        assert covered == {b.block_id for b in blocks}


def test_overlap_only_carries_blocks_that_were_new(cfg: AppConfig) -> None:
    """A block may appear in at most two chunks (its own and the next one's overlap)."""
    from collections import Counter

    chunks = fixed_tokens(_blocks_with_overlap_across_table(), "doc-x", cfg)
    occurrences = Counter(b for c in chunks for b in c.block_ids)
    assert max(occurrences.values()) <= 2
    # the chunk after the first table carries the page-2 overlap but is identified by its
    # first *new* block (page 3), so it sorts after the page-3 table chunk
    after_table = next(c for c in chunks if "blk-doc-x-p0003-0001" in c.block_ids)
    assert after_table.chunk_id.startswith("chunk-doc-x-p0003-")
    assert after_table.page_number == 2  # page span still reports the carried overlap


def test_chunk_ids_sort_for_documents_with_many_pages(cfg: AppConfig) -> None:
    blocks = [_block("doc-x", page, 0, f"Page {page} body text.") for page in range(1, 13)]
    chunks = by_page(blocks, "doc-x", cfg)
    ids = [c.chunk_id for c in chunks]
    assert ids == sorted(ids)
    assert ids[-1].endswith("p0012-0000")
