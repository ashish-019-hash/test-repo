"""Unit tests for AttributeExtractionAgent: full pipeline run against the real fixture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from doc_extractor.agents.attribute_extraction import AttributeExtractionAgent
from doc_extractor.chunking import chunk_document
from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import StageValidationError
from doc_extractor.ingest import load_document
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.schemas.attribute import Attribute
from doc_extractor.schemas.state import PipelineState, StageName, StageRecord


@dataclass
class _FakeProvider:
    name: str = "rules"

    def call(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("rules-mode provider must never be called by the scoring pipeline")


def _agent(cfg: AppConfig) -> AttributeExtractionAgent:
    return AttributeExtractionAgent(cfg, _FakeProvider())


@pytest.fixture(scope="module")
def loaded(cfg: AppConfig, telecom_spec_md: Path):
    doc = load_document(telecom_spec_md, cfg)
    chunks = chunk_document(doc, cfg)
    return doc, chunks


def _state(loaded) -> PipelineState:
    doc, chunks = loaded
    return {
        "document": doc,
        "chunks": chunks,
        "stages": {"chunking": StageRecord(stage="chunking", status="succeeded", attempts=1)},
    }


@pytest.fixture(scope="module")
def result(cfg: AppConfig, loaded):
    agent = _agent(cfg)
    return agent.run(_state(loaded))


def _by_name(attrs: list[Attribute], display_name: str) -> Attribute:
    matches = [a for a in attrs if a.display_name == display_name]
    assert len(matches) == 1, f"expected exactly one attribute named {display_name!r}, got {len(matches)}"
    return matches[0]


def test_run_succeeds(result):
    assert result["stages"]["attribute_extraction"].status == "succeeded"


def test_cir_worked_example(result):
    attributes = result["attributes"]
    cir = _by_name(attributes, "CIR")

    assert cir.attribute_name == "cir"
    assert cir.entity == "EVC"
    assert cir.binding.scope == "table_subject"
    assert cir.binding.confidence == pytest.approx(1.0)
    assert cir.layer == "CFS"
    assert cir.score.raw_sum == pytest.approx(1.50)
    assert cir.score.clamped == pytest.approx(1.0)
    assert cir.score.short_circuit is True
    assert cir.score.route == "accept"
    assert cir.confidence == pytest.approx(cir.score.clamped)

    assert cir.unit is not None
    assert cir.unit.raw == "Mbps"
    assert cir.unit.base == "bit/s"
    assert cir.unit.factor == pytest.approx(1e6)

    assert cir.value_domain is not None
    assert cir.value_domain.kind == "range"
    assert cir.value_domain.from_ == 10
    assert cir.value_domain.to == 1000

    assert cir.default == "100"
    assert cir.optionality == "M"
    assert cir.aliases == ["Committed Information Rate"]


def test_uni_is_promoted_not_an_attribute(result):
    attributes = result["attributes"]
    discarded = result["discarded_attributes"]

    assert not [a for a in attributes if a.display_name == "UNI"]
    uni_discards = [a for a in discarded if a.display_name == "UNI"]
    assert uni_discards
    assert all(a.score.discard_reason == "promoted" for a in uni_discards)
    assert all(a.score.route == "discard" for a in uni_discards)


def test_msisdn_is_a_review_item_bound_to_subscriber(result):
    attributes = result["attributes"]
    msisdn = _by_name(attributes, "MSISDN")
    assert msisdn.score.route == "review"
    assert msisdn.entity == "Subscriber"
    assert msisdn.binding.scope == "syntactic"

    review_items = result["human_review_queue"]
    assert any(item.payload.get("attribute_id") == msisdn.attribute_id for item in review_items)


def test_order_and_provisioning_are_discarded_below_threshold(result):
    discarded = result["discarded_attributes"]
    order = [a for a in discarded if a.display_name == "Order"]
    provisioning = [a for a in discarded if a.display_name == "Provisioning"]
    assert order and all(a.score.discard_reason == "below_threshold" for a in order)
    assert provisioning and all(a.score.discard_reason == "below_threshold" for a in provisioning)


def test_no_duplicate_attribute_ids_across_accepted_and_discarded(result):
    ids_ = [a.attribute_id for a in [*result["attributes"], *result["discarded_attributes"]]]
    assert len(ids_) == len(set(ids_))


def test_all_evidence_is_a_literal_substring_of_its_chunk(result, loaded):
    _doc, chunks = loaded
    chunk_by_id = {c.chunk_id: c for c in chunks}
    for attr in [*result["attributes"], *result["discarded_attributes"]]:
        for ev in attr.evidence:
            assert ev.text in chunk_by_id[ev.chunk_id].source_text


def test_confidence_matches_score_clamped_everywhere(result):
    for attr in [*result["attributes"], *result["discarded_attributes"]]:
        assert attr.confidence == pytest.approx(attr.score.clamped, abs=1e-9)


def test_evc_id_deduped_to_single_attribute(result):
    attributes = result["attributes"]
    matches = [a for a in attributes if a.attribute_name == "evcId" and a.entity == "EVC"]
    assert len(matches) == 1
    assert matches[0].score.route == "accept"


def test_rules_provider_never_invoked(cfg: AppConfig, loaded):
    # A provider that would raise if .call() is ever invoked; rules mode must not touch it.
    agent = _agent(cfg)
    agent.run(_state(loaded))  # would raise AssertionError inside _FakeProvider.call if reached


def test_validate_output_rejects_evidence_not_in_chunk(cfg: AppConfig, loaded):
    agent = _agent(cfg)
    doc, chunks = loaded
    state = _state(loaded)
    delta = agent.execute(state, TraceCollector(stage=StageName.attribute_extraction))
    bad_attr = delta["attributes"][0].model_copy(
        update={
            "evidence": [delta["attributes"][0].evidence[0].model_copy(update={"text": "not-in-any-chunk"})]
        }
    )
    bad_delta = {**delta, "attributes": [bad_attr, *delta["attributes"][1:]]}
    with pytest.raises(StageValidationError):
        agent.validate_output(bad_delta, state)
