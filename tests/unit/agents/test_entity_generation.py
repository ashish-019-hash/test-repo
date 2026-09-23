"""Tests for Agent 5: EntityGenerationAgent."""

from __future__ import annotations

from doc_extractor.agents.entity_generation import EntityGenerationAgent
from doc_extractor.exceptions import StageValidationError
from doc_extractor.schemas.state import StageName
from helpers_c import FakeProvider, base_state


def _run(cfg):
    agent = EntityGenerationAgent(cfg, FakeProvider("rules"))
    state = base_state(StageName.attribute_storage)
    delta = agent.run(state)
    return delta["entities"]


def test_evc_binding_entity_has_high_confidence_from_worked_example(cfg) -> None:
    entities = _run(cfg)
    evc = next(e for e in entities if e.entity_name == "EVC")
    assert evc.origin == "binding"
    assert evc.entity_type == "Service"
    assert evc.layer == "CFS"
    assert evc.confidence == 0.9  # 1 chunk + 5 attributes -> mentions=6 -> capped at min(1, 6/3)=1 * 0.9
    assert set(evc.source_attributes) == {
        "attr-cir",
        "attr-eir",
        "attr-evcid",
        "attr-servicetype",
        "attr-cosname",
    }


def test_uni_merges_binding_and_promoted_prose_chunk(cfg) -> None:
    entities = _run(cfg)
    uni = next(e for e in entities if e.entity_name == "UNI")
    assert uni.origin == "binding"  # binding beats promotion in the merge priority
    # source_chunks should include both the table chunk (binding) and the prose chunk
    # (promotion), because the merge unions chunks/attrs/evidence across sources.
    chunk_pages = {c for c in uni.source_chunks}
    assert len(chunk_pages) == 2


def test_subscriber_entity_from_syntactic_binding(cfg) -> None:
    entities = _run(cfg)
    subscriber = next(e for e in entities if e.entity_name == "Subscriber")
    assert subscriber.entity_type == "Party"
    assert set(subscriber.source_attributes) == {"attr-msisdn", "attr-imsi"}


def test_order_is_not_promoted_to_an_entity(cfg) -> None:
    """ "Order" appears once in prose (not >= min_mentions distinct chunks) and is not bound
    by any attribute or promoted -> must not become an entity."""
    entities = _run(cfg)
    names = {e.entity_name for e in entities}
    assert "Order" not in names


def test_dangling_attribute_does_not_produce_an_entity(cfg) -> None:
    entities = _run(cfg)
    for e in entities:
        assert "attr-provisioningtime" not in e.source_attributes


def test_every_entity_has_evidence_and_is_grounded(cfg) -> None:
    entities = _run(cfg)
    assert entities  # sanity: something was generated
    for e in entities:
        assert e.supporting_evidence
        assert e.source_chunks


def test_entity_ids_are_unique_and_sorted(cfg) -> None:
    entities = _run(cfg)
    ids_ = [e.entity_id for e in entities]
    assert len(ids_) == len(set(ids_))
    assert ids_ == sorted(ids_)


def test_empty_attributes_is_a_valid_input(cfg) -> None:
    """`validate_input` must allow an empty attributes list (a document with no attributes
    at all can still surface lexicon-derived entities)."""
    agent = EntityGenerationAgent(cfg, FakeProvider("rules"))
    state = base_state(StageName.attribute_storage, attributes=[], discarded_attributes=[])
    delta = agent.run(state)
    assert isinstance(delta["entities"], list)


def test_missing_chunks_fails_validation(cfg) -> None:
    agent = EntityGenerationAgent(cfg, FakeProvider("rules"))
    state = base_state(StageName.attribute_storage, chunks=[])
    try:
        agent.run(state)
        raised = False
    except StageValidationError:
        raised = True
    assert raised
