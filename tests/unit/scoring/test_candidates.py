"""Unit tests for `scoring.candidates.generate_candidates` against the real ingest+chunking
pipeline output for tests/fixtures/telecom_spec.md."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_extractor.chunking import chunk_document
from doc_extractor.config.models import AppConfig
from doc_extractor.ingest import load_document
from doc_extractor.scoring.candidates import generate_candidates
from doc_extractor.scoring.lexicons import Lexicons, load_lexicons


@pytest.fixture(scope="module")
def lex(cfg: AppConfig) -> Lexicons:
    return load_lexicons(cfg)


@pytest.fixture(scope="module")
def loaded(cfg: AppConfig, telecom_spec_md: Path):
    doc = load_document(telecom_spec_md, cfg)
    chunks = chunk_document(doc, cfg)
    return doc, chunks


@pytest.fixture(scope="module")
def all_candidates(cfg, lex, loaded):
    doc, chunks = loaded
    out = []
    for chunk in chunks:
        out.extend((chunk, c) for c in generate_candidates(chunk, doc, lex, cfg))
    return out


def test_cir_spec_table_candidate_has_all_cells(all_candidates):
    matches = [c for _chunk, c in all_candidates if c.raw_name == "CIR" and c.origin == "spec_table"]
    assert len(matches) == 1
    cand = matches[0]
    assert cand.cells["type"] == "Integer"
    assert cand.cells["units"] == "Mbps"
    assert cand.cells["optionality"] == "M"
    assert cand.cells["range"] == "10–1000"
    assert cand.cells["default"] == "100"
    assert cand.cells["description"] == "Committed Information Rate for the EVC"


def test_all_five_evc_table_rows_generated(all_candidates):
    names = {c.raw_name for _chunk, c in all_candidates if c.origin == "spec_table"}
    assert {
        "CIR",
        "EIR",
        "EVC ID",
        "Service Type",
        "CoS Name",
        "Port Speed",
        "Physical Medium",
        "MAC Address",
    } <= names


def test_revision_history_table_produces_no_spec_table_candidates(all_candidates):
    # Header (Version, Date, Author, Change) matches none of cfg.scoring.spec_table_headers.
    names = {c.raw_name for _chunk, c in all_candidates if c.origin == "spec_table"}
    assert "1.0" not in names
    assert "J. Smith" not in names


def test_uni_and_order_and_msisdn_prose_candidates_present(all_candidates):
    prose_names = {c.raw_name for _chunk, c in all_candidates if c.origin == "prose_regex"}
    assert "UNI" in prose_names
    assert "Order" in prose_names
    assert "MSISDN" in prose_names
    assert "IMSI" in prose_names
    assert "Provisioning" in prose_names


def test_msisdn_sentence_spans_both_source_sentences(all_candidates):
    matches = [c for _chunk, c in all_candidates if c.raw_name == "MSISDN"]
    assert len(matches) == 1
    sentence = matches[0].sentence
    assert sentence is not None
    assert "IMSI" in sentence
    assert "+447700900123" in sentence


def test_furniture_blocks_never_produce_candidates(loaded, all_candidates):
    doc, _chunks = loaded
    furniture_block_ids = {b.block_id for b in doc.blocks if b.kind == "furniture"}
    for _chunk, cand in all_candidates:
        assert cand.block_id not in furniture_block_ids


def test_every_candidate_source_text_is_substring_of_its_chunk(all_candidates):
    for chunk, cand in all_candidates:
        assert cand.source_text == chunk.source_text
        if cand.sentence is not None:
            assert cand.sentence in chunk.source_text
