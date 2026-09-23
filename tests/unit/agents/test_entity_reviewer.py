"""Tests for Agent 7: EntityReviewerAgent."""

from __future__ import annotations

from doc_extractor.agents.entity_generation import EntityGenerationAgent
from doc_extractor.agents.entity_normalization import EntityNormalizationAgent
from doc_extractor.agents.entity_reviewer import EntityReviewerAgent
from doc_extractor.schemas.state import StageName, StageRecord
from helpers_c import FakeProvider, base_state


def _state_through_normalization(cfg):
    provider = FakeProvider("rules")
    state = base_state(StageName.attribute_storage)
    gen = EntityGenerationAgent(cfg, provider)
    state.update(gen.run(state))
    state["stages"][str(StageName.entity_generation)] = StageRecord(
        stage=StageName.entity_generation, attempts=1, status="succeeded"
    )
    norm = EntityNormalizationAgent(cfg, provider)
    state.update(norm.run(state))
    state["stages"][str(StageName.entity_normalization)] = StageRecord(
        stage=StageName.entity_normalization, attempts=1, status="succeeded"
    )
    return state


def _run_reviewer(cfg):
    state = _state_through_normalization(cfg)
    agent = EntityReviewerAgent(cfg, FakeProvider("rules"))
    delta = agent.run(state)
    state.update(delta)
    return state


def test_every_original_entity_belongs_to_exactly_one_group(cfg) -> None:
    state = _run_reviewer(cfg)
    entities = state["entities"]
    groups = state["duplicate_groups"]
    all_members: list[str] = []
    for g in groups:
        all_members.extend(g.member_ids)
    assert sorted(all_members) == sorted(e.entity_id for e in entities)
    assert len(all_members) == len(set(all_members))  # no entity in two groups


def test_canonical_entities_match_groups_one_to_one(cfg) -> None:
    state = _run_reviewer(cfg)
    groups = state["duplicate_groups"]
    canonicals = state["canonical_entities"]
    assert {g.canonical_id for g in groups} == {c.canonical_id for c in canonicals}


def test_review_decisions_cover_every_canonical_entity_exactly_once(cfg) -> None:
    state = _run_reviewer(cfg)
    canonicals = state["canonical_entities"]
    decisions = state["review_decisions"]
    assert {d.entity_id for d in decisions} == {c.canonical_id for c in canonicals}
    assert len(decisions) == len(canonicals)


def test_evc_and_uni_and_subscriber_are_accepted(cfg) -> None:
    state = _run_reviewer(cfg)
    decisions_by_name = {
        c.canonical_name: d
        for c in state["canonical_entities"]
        for d in state["review_decisions"]
        if d.entity_id == c.canonical_id
    }
    for name in ("EVC", "UNI", "Subscriber"):
        assert decisions_by_name[name].validation_status == "ACCEPT", (
            f"{name}: {decisions_by_name[name].reason}"
        )


def test_no_duplicate_groups_are_created_for_distinct_entities(cfg) -> None:
    """Service vs. Service Provider must remain distinct canonical entities."""
    state = _run_reviewer(cfg)
    names = {c.canonical_name for c in state["canonical_entities"]}
    if "Service" in names and "Service Provider" in names:
        assert names.issuperset({"Service", "Service Provider"})
    groups_with_multiple_members = [g for g in state["duplicate_groups"] if len(g.member_ids) > 1]
    assert groups_with_multiple_members == []  # nothing in this fixture is a true duplicate
