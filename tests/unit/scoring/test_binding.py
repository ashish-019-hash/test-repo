"""Unit tests for `scoring.binding`: scope priority, confidence formula, dangling."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_extractor.chunking import chunk_document
from doc_extractor.config.models import AppConfig
from doc_extractor.ingest import load_document
from doc_extractor.schemas.attribute import AttributeCandidate
from doc_extractor.scoring.binding import (
    bind_entity,
    heading_implied_entity,
    known_entities_for_document,
    table_subject_entities,
)
from doc_extractor.scoring.candidates import generate_candidates
from doc_extractor.scoring.lexicons import Lexicons, load_lexicons
from doc_extractor.scoring.signals import ScoringContext


@pytest.fixture(scope="module")
def lex(cfg: AppConfig) -> Lexicons:
    return load_lexicons(cfg)


@pytest.fixture(scope="module")
def loaded(cfg: AppConfig, telecom_spec_md: Path):
    doc = load_document(telecom_spec_md, cfg)
    chunks = chunk_document(doc, cfg)
    return doc, chunks


@pytest.fixture(scope="module")
def known_entities(loaded, lex):
    doc, _chunks = loaded
    return known_entities_for_document(doc, lex)


def _find(cfg, lex, doc, chunks, raw_name: str, origin: str | None = None):
    for chunk in chunks:
        for cand in generate_candidates(chunk, doc, lex, cfg):
            if cand.raw_name == raw_name and (origin is None or cand.origin == origin):
                return chunk, cand
    raise AssertionError(f"no candidate named {raw_name!r} (origin={origin})")


def _ctx(doc, chunk, cand, known_entities, **overrides) -> ScoringContext:
    block = doc.block_map().get(cand.block_id)
    base = dict(
        candidate=cand,
        chunk=chunk,
        block=block,
        heading_path=list(chunk.heading_path),
        table=block.table if block else None,
        sentence=cand.sentence,
        document_title=doc.title,
        known_entities=known_entities,
    )
    base.update(overrides)
    return ScoringContext(**base)


# ---- heading_implied_entity ------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading, expected",
    [
        ("4.3 EVC Service Attributes", "EVC"),
        ("5.1 UNI Attributes", "UNI"),
        ("4 Service Attributes", "Service"),
        ("6 Subscriber", None),
        ("Introduction", None),
    ],
)
def test_heading_implied_entity(heading: str, expected: str | None) -> None:
    assert heading_implied_entity(heading) == expected


# ---- scope 1: table_subject -------------------------------------------------------------------


def test_cir_binds_via_table_subject(cfg, lex, loaded, known_entities):
    doc, chunks = loaded
    chunk, cand = _find(cfg, lex, doc, chunks, "CIR", "spec_table")
    ctx = _ctx(
        doc, chunk, cand, known_entities, table_subjects=table_subject_entities(doc, known_entities, lex)
    )
    binding = bind_entity(ctx, lex, cfg)
    assert binding.scope == "table_subject"
    assert binding.entity_name == "EVC"
    assert binding.confidence == pytest.approx(1.0)
    assert binding.structural_distance == 0


def test_uni_table_rows_bind_via_table_subject(cfg, lex, loaded, known_entities):
    doc, chunks = loaded
    chunk, cand = _find(cfg, lex, doc, chunks, "Port Speed", "spec_table")
    ctx = _ctx(doc, chunk, cand, known_entities)
    binding = bind_entity(ctx, lex, cfg)
    assert binding.scope == "table_subject"
    assert binding.entity_name == "UNI"


# ---- scope 3: syntactic (nearest_heading is intentionally too weak here) ----------------------


def test_msisdn_binds_via_syntactic_genitive(cfg, lex, loaded, known_entities):
    doc, chunks = loaded
    chunk, cand = _find(cfg, lex, doc, chunks, "MSISDN", "prose_regex")
    ctx = _ctx(doc, chunk, cand, known_entities)
    binding = bind_entity(ctx, lex, cfg)
    assert binding.scope == "syntactic"
    assert binding.entity_name == "Subscriber"
    assert binding.confidence == pytest.approx(0.85)


# ---- scope 4: compound_modifier ----------------------------------------------------------------


def test_compound_modifier_binds_evc_id_to_evc(cfg, lex, loaded, known_entities):
    cand = AttributeCandidate(
        raw_name="EVC ID",
        chunk_id="chunk-x",
        block_id="blk-x",
        source_text="EVC ID",
        origin="prose_regex",
    )
    ctx = ScoringContext(
        candidate=cand,
        chunk=None,
        block=None,
        heading_path=[],
        table=None,
        sentence=None,
        document_title=None,
        known_entities=known_entities,
    )
    binding = bind_entity(ctx, lex, cfg)
    assert binding.scope == "compound_modifier"
    assert binding.entity_name == "EVC"
    assert binding.confidence == pytest.approx(0.7)


# ---- scope 5 / dangling -------------------------------------------------------------------------


def test_document_default_used_when_configured(cfg, lex):
    cfg2 = cfg.model_copy(
        update={"binding": cfg.binding.model_copy(update={"document_default_entity": "EVC"})}
    )
    cand = AttributeCandidate(
        raw_name="Zorblatt",
        chunk_id="chunk-x",
        block_id="blk-x",
        source_text="Zorblatt",
        origin="prose_regex",
    )
    ctx = ScoringContext(
        candidate=cand,
        chunk=None,
        block=None,
        heading_path=[],
        table=None,
        sentence=None,
        document_title="Some Document",
        known_entities={},
    )
    binding = bind_entity(ctx, lex, cfg2)
    assert binding.scope == "document_default"
    assert binding.entity_name == "EVC"
    assert binding.confidence == pytest.approx(0.5)


def test_dangling_when_nothing_matches(cfg, lex):
    cand = AttributeCandidate(
        raw_name="Zorblatt",
        chunk_id="chunk-x",
        block_id="blk-x",
        source_text="Zorblatt",
        origin="prose_regex",
    )
    ctx = ScoringContext(
        candidate=cand,
        chunk=None,
        block=None,
        heading_path=[],
        table=None,
        sentence=None,
        document_title=None,
        known_entities={},
    )
    binding = bind_entity(ctx, lex, cfg)
    assert binding.scope == "dangling"
    assert binding.entity_name is None
    assert binding.confidence == pytest.approx(0.0)


def test_table_subject_entities_includes_evc_and_uni(cfg, lex, loaded, known_entities):
    doc, _chunks = loaded
    subjects = table_subject_entities(doc, known_entities, lex)
    assert "evc" in subjects
    assert "uni" in subjects
