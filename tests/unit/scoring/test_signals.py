"""Unit tests for the deterministic scoring signals (structural/lexical/value_domain/
syntactic/negative) using the real telecom_spec fixture as ground truth, plus small
synthetic ScoringContexts for cases the fixture doesn't exercise directly."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_extractor.chunking import chunk_document
from doc_extractor.config.models import AppConfig
from doc_extractor.ingest import load_document
from doc_extractor.schemas.attribute import AttributeCandidate
from doc_extractor.scoring import lexical, negative, structural, syntactic, value_domain
from doc_extractor.scoring.binding import known_entities_for_document
from doc_extractor.scoring.candidates import generate_candidates
from doc_extractor.scoring.lexicons import Lexicons, load_lexicons
from doc_extractor.scoring.signals import (
    ScoringContext,
    build_registry,
    head_noun,
    merged_span_for,
    split_sentences,
)


@pytest.fixture(scope="module")
def lex(cfg: AppConfig) -> Lexicons:
    return load_lexicons(cfg)


@pytest.fixture(scope="module")
def loaded(cfg: AppConfig, telecom_spec_md: Path):
    doc = load_document(telecom_spec_md, cfg)
    chunks = chunk_document(doc, cfg)
    return doc, chunks


@pytest.fixture(scope="module")
def all_records(cfg: AppConfig, lex: Lexicons, loaded):
    doc, chunks = loaded
    records = []
    for chunk in chunks:
        for cand in generate_candidates(chunk, doc, lex, cfg):
            records.append((chunk, cand))
    return doc, chunks, records


def _ctx_for(doc, lex, cfg, chunk, cand, **overrides) -> ScoringContext:
    block = doc.block_map().get(cand.block_id)
    known_entities = known_entities_for_document(doc, lex)
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


def _find(all_records, raw_name: str, origin: str | None = None):
    _doc, _chunks, records = all_records
    for chunk, cand in records:
        if cand.raw_name == raw_name and (origin is None or cand.origin == origin):
            return chunk, cand
    raise AssertionError(f"no candidate named {raw_name!r} (origin={origin})")


# ---- structural.py -------------------------------------------------------------------------


def test_spec_table_row_fires_on_cir(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "CIR", "spec_table")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    hit = structural.spec_table_row(ctx, lex, cfg)
    assert hit is not None
    assert hit.name == "structural.spec_table_row"
    assert hit.weight == pytest.approx(0.45)


def test_spec_table_row_does_not_fire_for_prose(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "Order", "prose_regex")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    assert structural.spec_table_row(ctx, lex, cfg) is None


def test_attribute_heading_fires_for_cir_heading_path(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "CIR", "spec_table")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    hit = structural.attribute_heading(ctx, lex, cfg)
    assert hit is not None
    assert hit.weight == pytest.approx(0.20)


# ---- lexical.py -----------------------------------------------------------------------------


def test_property_noun_head_fires_via_alias_expansion_for_cir(cfg, lex, loaded, all_records):
    """CIR itself has no property-noun head, but its gazetteer alias 'Committed
    Information Rate' does ('rate') -- the worked example's specific note."""
    doc, chunks = loaded
    chunk, cand = _find(all_records, "CIR", "spec_table")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    hit = lexical.property_noun_head(ctx, lex, cfg)
    assert hit is not None
    assert "rate" in hit.evidence.lower()


def test_gazetteer_hit_fires_for_cir(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "CIR", "spec_table")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    assert lexical.gazetteer_hit(ctx, lex, cfg) is not None


def test_gazetteer_hit_fires_for_msisdn(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "MSISDN", "prose_regex")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    assert lexical.gazetteer_hit(ctx, lex, cfg) is not None


# ---- value_domain.py --------------------------------------------------------------------------


def test_typed_value_fires_via_type_cell_for_cir(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "CIR", "spec_table")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    hit = value_domain.typed_value(ctx, lex, cfg)
    assert hit is not None
    assert "Integer" in hit.evidence


def test_typed_value_fires_via_e164_for_msisdn(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "MSISDN", "prose_regex")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    hit = value_domain.typed_value(ctx, lex, cfg)
    assert hit is not None
    assert "E.164" in hit.evidence
    assert "+447700900123" in hit.evidence


def test_value_domain_does_not_leak_across_table_rows(cfg, lex, loaded, all_records):
    """EVC ID has no range/default of its own; scanning must not pick up CIR's '10-1000'."""
    doc, chunks = loaded
    chunk, cand = _find(all_records, "EVC ID", "spec_table")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    assert value_domain.enum_range_default(ctx, lex, cfg) is None


def test_cardinality_marker_fires_via_optionality_cell(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "CIR", "spec_table")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    hit = value_domain.cardinality_marker(ctx, lex, cfg)
    assert hit is not None
    assert hit.weight == pytest.approx(0.15)


# ---- syntactic.py -----------------------------------------------------------------------------


def test_find_binding_pattern_genitive():
    found = syntactic.find_binding_pattern(
        "The Subscriber's MSISDN is an E.164 number such as +447700900123.",
        "MSISDN",
        ["Subscriber", "EVC", "UNI"],
    )
    assert found is not None
    entity, evidence, kind = found
    assert entity == "Subscriber"
    assert kind == "genitive"
    assert "MSISDN" in evidence


def test_find_binding_pattern_rfc2119():
    found = syntactic.find_binding_pattern(
        "Each UNI SHALL have a physical medium and a MAC address.", "MAC address", ["UNI", "Subscriber"]
    )
    assert found is not None
    entity, _evidence, kind = found
    assert entity == "UNI"
    assert kind == "rfc2119"


def test_split_rfc2119_tail():
    assert syntactic.split_rfc2119_tail("an IMSI and an MSISDN") == ["IMSI", "MSISDN"]


def test_binding_pattern_signal_fires_for_msisdn(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "MSISDN", "prose_regex")
    known_entities = known_entities_for_document(doc, lex)
    ctx = _ctx_for(doc, lex, cfg, chunk, cand, known_entities=known_entities)
    hit = syntactic.binding_pattern(ctx, lex, cfg)
    assert hit is not None
    assert hit.weight == pytest.approx(0.20)


def test_binding_pattern_does_not_fire_for_spec_table_origin(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "CIR", "spec_table")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    assert syntactic.binding_pattern(ctx, lex, cfg) is None


# ---- negative.py ------------------------------------------------------------------------------


def test_lifecycle_subject_fires_for_order(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "Order", "prose_regex")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    hit = negative.lifecycle_subject(ctx, lex, cfg)
    assert hit is not None
    assert hit.weight == pytest.approx(-0.30)


def test_verb_nominalisation_fires_for_provisioning(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "Provisioning", "prose_regex")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand)
    hit = negative.verb_nominalisation(ctx, lex, cfg)
    assert hit is not None
    assert hit.weight == pytest.approx(-0.25)


def test_verb_nominalisation_does_not_fire_for_property_noun_head(cfg, lex):
    # "Condition" ends in -tion but its head noun IS a configured property noun
    # (property_nouns.yaml status list), so the negative signal must not fire.
    cand = AttributeCandidate(
        raw_name="Condition",
        chunk_id="chunk-x",
        block_id="blk-x",
        source_text="Condition",
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
    )
    assert negative.verb_nominalisation(ctx, lex, cfg) is None


def test_has_sub_attributes_fires_from_bind_counts(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "UNI", "prose_regex")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand, bind_counts={"uni": 3})
    hit = negative.has_sub_attributes(ctx, lex, cfg)
    assert hit is not None
    assert hit.weight == pytest.approx(-0.40)


def test_has_sub_attributes_does_not_fire_below_threshold(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    chunk, cand = _find(all_records, "UNI", "prose_regex")
    ctx = _ctx_for(doc, lex, cfg, chunk, cand, bind_counts={"uni": 1})
    assert negative.has_sub_attributes(ctx, lex, cfg) is None


def test_counted_instances_fires(cfg, lex):
    cand = AttributeCandidate(
        raw_name="UNI",
        chunk_id="chunk-x",
        block_id="blk-x",
        source_text="There are 12 such UNIs in the reference deployment.",
        origin="prose_regex",
        sentence="There are 12 such UNIs in the reference deployment.",
    )
    ctx = ScoringContext(
        candidate=cand,
        chunk=None,
        block=None,
        heading_path=[],
        table=None,
        sentence=cand.sentence,
        document_title=None,
    )
    hit = negative.counted_instances(ctx, lex, cfg)
    assert hit is not None
    assert hit.weight == pytest.approx(-0.25)


def test_furniture_fires_on_furniture_block(cfg, lex, loaded, all_records):
    doc, chunks = loaded
    # "Page 3 of 3" is a real furniture block in the fixture.
    furniture_block = next(b for b in doc.blocks if b.kind == "furniture" and "page" in b.text.lower())
    cand = AttributeCandidate(
        raw_name=furniture_block.text,
        chunk_id="chunk-x",
        block_id=furniture_block.block_id,
        source_text=furniture_block.text,
        origin="prose_regex",
    )
    ctx = ScoringContext(
        candidate=cand,
        chunk=None,
        block=furniture_block,
        heading_path=[],
        table=None,
        sentence=None,
        document_title=None,
    )
    hit = negative.furniture(ctx, lex, cfg)
    assert hit is not None
    assert hit.weight == pytest.approx(-1.0)


# ---- signals.py helpers -----------------------------------------------------------------------


def test_head_noun_strips_parenthetical():
    assert head_noun("Committed Information Rate (CIR)") == "rate"


def test_split_sentences_basic():
    sents = split_sentences("A first sentence. A second one. And a third.")
    assert sents == ["A first sentence.", "A second one.", "And a third."]


def test_merged_span_for_merges_all_matching_sentences():
    text = "Each Subscriber SHALL have an IMSI and an MSISDN. The Subscriber's MSISDN is an E.164 number such as +447700900123. The Order is created."
    span = merged_span_for(text, "MSISDN")
    assert span is not None
    assert span in text
    assert "IMSI" in span
    assert "+447700900123" in span
    assert "Order" not in span


def test_merged_span_for_returns_none_when_absent():
    assert merged_span_for("No such term here.", "MSISDN") is None


def test_build_registry_order_and_length():
    registry = build_registry()
    assert len(registry) == 14
