"""Unit tests for `scoring.engine.score`: raw_sum, clamping, short-circuit, routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_extractor.chunking import chunk_document
from doc_extractor.config.models import AppConfig
from doc_extractor.ingest import load_document
from doc_extractor.scoring.candidates import generate_candidates
from doc_extractor.scoring.engine import score
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


def _cir_ctx(cfg, lex, loaded) -> ScoringContext:
    doc, chunks = loaded
    block_map = doc.block_map()
    for chunk in chunks:
        for cand in generate_candidates(chunk, doc, lex, cfg):
            if cand.raw_name == "CIR" and cand.origin == "spec_table":
                block = block_map.get(cand.block_id)
                return ScoringContext(
                    candidate=cand,
                    chunk=chunk,
                    block=block,
                    heading_path=list(chunk.heading_path),
                    table=block.table if block else None,
                    sentence=cand.sentence,
                    document_title=doc.title,
                )
    raise AssertionError("CIR spec_table candidate not found")


def test_cir_worked_example_exact_breakdown(cfg, lex, loaded):
    ctx = _cir_ctx(cfg, lex, loaded)
    breakdown = score(ctx, lex, cfg)

    fired = {h.name: h.weight for h in breakdown.hits}
    assert fired == {
        "structural.spec_table_row": pytest.approx(0.45),
        "structural.attribute_heading": pytest.approx(0.20),
        "lexical.property_noun_head": pytest.approx(0.20),
        "lexical.gazetteer_hit": pytest.approx(0.15),
        "value_domain.typed_value": pytest.approx(0.20),
        "value_domain.enum_range_default": pytest.approx(0.15),
        "value_domain.cardinality_marker": pytest.approx(0.15),
    }
    assert breakdown.raw_sum == pytest.approx(1.50)
    assert breakdown.clamped == pytest.approx(1.0)
    assert breakdown.short_circuit is True
    assert breakdown.route == "accept"
    assert breakdown.discard_reason is None


def test_clamp_never_exceeds_one_without_short_circuit(cfg, lex):
    from doc_extractor.schemas.attribute import AttributeCandidate

    # A kv_pair candidate cannot short-circuit (requires spec_table_row); pile on enough
    # positive signals to exceed 1.0 raw and confirm clamped stops at 1.0.
    cand = AttributeCandidate(
        raw_name="Bandwidth",
        chunk_id="chunk-x",
        block_id="blk-x",
        source_text="Bandwidth: 10 Mbps, one of [0..1] range, default 5",
        origin="kv_pair",
        cells={"description": "10 Mbps, one of [0..1] range, default 5"},
    )
    ctx = ScoringContext(
        candidate=cand,
        chunk=None,
        block=None,
        heading_path=[],
        table=None,
        sentence=None,
        document_title=None,
    )
    breakdown = score(ctx, lex, cfg)
    assert breakdown.clamped <= 1.0
    assert breakdown.clamped >= 0.0


def test_routing_boundaries(cfg, lex):
    from doc_extractor.schemas.attribute import AttributeCandidate

    # Below review threshold (0.50): only structural.key_value_pair (0.30) fires.
    cand = AttributeCandidate(
        raw_name="Xyzzy123",
        chunk_id="chunk-x",
        block_id="blk-x",
        source_text="irrelevant",
        origin="kv_pair",
    )
    ctx = ScoringContext(
        candidate=cand,
        chunk=None,
        block=None,
        heading_path=[],
        table=None,
        sentence=None,
        document_title=None,
    )
    breakdown = score(ctx, lex, cfg)
    assert breakdown.route == "discard"
    assert breakdown.discard_reason == "below_threshold"


def test_discard_reason_promoted_when_has_sub_attributes_fires(cfg, lex):
    from doc_extractor.schemas.attribute import AttributeCandidate

    cand = AttributeCandidate(
        raw_name="UNI",
        chunk_id="chunk-x",
        block_id="blk-x",
        source_text="The UNI has attributes.",
        origin="prose_regex",
        sentence="The UNI has attributes.",
    )
    ctx = ScoringContext(
        candidate=cand,
        chunk=None,
        block=None,
        heading_path=[],
        table=None,
        sentence=cand.sentence,
        document_title=None,
        bind_counts={"uni": 5},
    )
    breakdown = score(ctx, lex, cfg)
    assert breakdown.route == "discard"
    assert breakdown.discard_reason == "promoted"


def test_short_circuit_disabled_by_negative_signal(cfg, lex):
    from doc_extractor.schemas.attribute import AttributeCandidate

    # A spec_table row with >=3 populated cells but also a furniture match must not
    # short-circuit to full confidence.
    cand = AttributeCandidate(
        raw_name="Figure 2",
        chunk_id="chunk-x",
        block_id="blk-x",
        source_text="irrelevant",
        origin="spec_table",
        cells={"type": "String", "units": "n/a", "optionality": "M", "default": "x", "description": "d"},
    )
    ctx = ScoringContext(
        candidate=cand,
        chunk=None,
        block=None,
        heading_path=[],
        table=None,
        sentence=None,
        document_title=None,
    )
    breakdown = score(ctx, lex, cfg)
    assert breakdown.short_circuit is False
