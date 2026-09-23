"""Tests for Agent 6: EntityNormalizationAgent."""

from __future__ import annotations

import pytest

from doc_extractor.agents.entity_generation import EntityGenerationAgent
from doc_extractor.agents.entity_normalization import EntityNormalizationAgent
from doc_extractor.exceptions import StageValidationError
from doc_extractor.schemas.entity import Entity
from doc_extractor.schemas.state import StageName, StageRecord
from helpers_c import FakeProvider, base_state


def _generated_entities(cfg):
    agent = EntityGenerationAgent(cfg, FakeProvider("rules"))
    state = base_state(StageName.attribute_storage)
    return agent.run(state)["entities"]


def _state_with_entities(cfg, entities):
    state = base_state(StageName.attribute_storage, entities=entities)
    state["stages"][str(StageName.entity_generation)] = StageRecord(
        stage=StageName.entity_generation, attempts=1, status="succeeded"
    )
    return state


def test_every_entity_gets_exactly_one_normalized_record(cfg) -> None:
    entities = _generated_entities(cfg)
    agent = EntityNormalizationAgent(cfg, FakeProvider("rules"))
    state = _state_with_entities(cfg, entities)
    delta = agent.run(state)
    normalized = delta["normalized_entities"]
    assert len(normalized) == len(entities)
    assert {n.entity_id for n in normalized} == {e.entity_id for e in entities}


def test_normalization_lowercases_and_records_reasoning(cfg) -> None:
    entities = _generated_entities(cfg)
    agent = EntityNormalizationAgent(cfg, FakeProvider("rules"))
    state = _state_with_entities(cfg, entities)
    delta = agent.run(state)
    normalized = delta["normalized_entities"]
    evc = next(n for n in normalized if n.original_entity_name == "EVC")
    assert evc.normalized_entity_name == "evc"
    assert evc.reasoning  # non-empty explanation string


def test_no_change_reasoning_when_already_normalized(cfg) -> None:
    entity = Entity(
        entity_id="ent-lowercase",
        entity_name="evc",
        entity_type="Service",  # type: ignore[arg-type]
        layer="CFS",
        source_attributes=[],
        source_chunks=["chunk-1"],
        supporting_evidence=[],
        origin="binding",  # type: ignore[arg-type]
        confidence=0.9,
    )
    agent = EntityNormalizationAgent(cfg, FakeProvider("rules"))
    state = _state_with_entities(cfg, [entity])
    delta = agent.run(state)
    (normalized,) = delta["normalized_entities"]
    assert normalized.normalized_entity_name == "evc"
    assert normalized.reasoning == "no change"


def test_missing_entities_fails_validation(cfg) -> None:
    agent = EntityNormalizationAgent(cfg, FakeProvider("rules"))
    state = _state_with_entities(cfg, [])
    del state["entities"]
    with pytest.raises(StageValidationError, match="'entities' is missing"):
        agent.run(state)


def test_empty_entities_normalize_to_empty(cfg) -> None:
    agent = EntityNormalizationAgent(cfg, FakeProvider("rules"))
    delta = agent.run(_state_with_entities(cfg, []))
    assert delta["normalized_entities"] == []
