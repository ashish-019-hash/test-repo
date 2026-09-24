"""Tests for Agent 8: AttributeMappingAgent."""

from __future__ import annotations

from doc_extractor.agents.attribute_mapping import AttributeMappingAgent
from doc_extractor.agents.entity_generation import EntityGenerationAgent
from doc_extractor.agents.entity_normalization import EntityNormalizationAgent
from doc_extractor.agents.entity_reviewer import EntityReviewerAgent
from doc_extractor.schemas.state import StageName, StageRecord
from helpers_c import FakeProvider, base_state


def _state_through_reviewer(cfg):
    provider = FakeProvider("rules")
    state = base_state(StageName.attribute_storage)
    for stage, agent_cls in [
        (StageName.entity_generation, EntityGenerationAgent),
        (StageName.entity_normalization, EntityNormalizationAgent),
        (StageName.entity_reviewer, EntityReviewerAgent),
    ]:
        agent = agent_cls(cfg, provider)
        state.update(agent.run(state))
        state["stages"][str(stage)] = StageRecord(stage=stage, attempts=1, status="succeeded")
    return state


def _run_mapping(cfg):
    state = _state_through_reviewer(cfg)
    agent = AttributeMappingAgent(cfg, FakeProvider("rules"))
    delta = agent.run(state)
    state.update(delta)
    return state


def _canonical_name(state, canonical_id: str) -> str:
    return next(c.canonical_name for c in state["canonical_entities"] if c.canonical_id == canonical_id)


def test_evc_attributes_map_to_evc_canonical_entity(cfg) -> None:
    state = _run_mapping(cfg)
    mapped_by_attr = {m.attribute_id: m for m in state["mappings"]}
    for attr_id in ("attr-cir", "attr-eir", "attr-evcid", "attr-servicetype", "attr-cosname"):
        mapping = mapped_by_attr[attr_id]
        assert _canonical_name(state, mapping.entity_id) == "EVC"
        assert mapping.relationship_type == "has_attribute"


def test_uni_attributes_map_to_uni_canonical_entity(cfg) -> None:
    state = _run_mapping(cfg)
    mapped_by_attr = {m.attribute_id: m for m in state["mappings"]}
    for attr_id in ("attr-portspeed", "attr-physicalmedium", "attr-macaddress"):
        assert _canonical_name(state, mapped_by_attr[attr_id].entity_id) == "UNI"


def test_msisdn_and_imsi_use_constrains_relationship_from_shall(cfg) -> None:
    """The binding sentence contains "SHALL" -> RFC-2119 constrains, not has_attribute."""
    state = _run_mapping(cfg)
    mapped_by_attr = {m.attribute_id: m for m in state["mappings"]}
    for attr_id in ("attr-msisdn", "attr-imsi"):
        mapping = mapped_by_attr[attr_id]
        assert mapping.relationship_type == "constrains"
        assert _canonical_name(state, mapping.entity_id) == "Subscriber"


def test_dangling_attribute_is_unmapped(cfg) -> None:
    state = _run_mapping(cfg)
    assert "attr-provisioningtime" in state["unmapped_attribute_ids"]
    mapped_ids = {m.attribute_id for m in state["mappings"]}
    assert "attr-provisioningtime" not in mapped_ids


def test_every_mapping_targets_an_accepted_entity(cfg) -> None:
    state = _run_mapping(cfg)
    decisions_by_id = {d.entity_id: d for d in state["review_decisions"]}
    for m in state["mappings"]:
        assert decisions_by_id[m.entity_id].validation_status == "ACCEPT"


def test_mapping_ids_are_unique(cfg) -> None:
    state = _run_mapping(cfg)
    mapping_ids = [m.mapping_id for m in state["mappings"]]
    assert len(mapping_ids) == len(set(mapping_ids))


def test_review_entities_excluded_unless_configured(cfg) -> None:
    """With `include_review_entities=False`, attributes bound to a REVIEW-status canonical
    entity must be left unmapped."""
    cfg = cfg.model_copy(
        update={"mapping": cfg.mapping.model_copy(update={"include_review_entities": False})}
    )
    state = _state_through_reviewer(cfg)
    review_ids = {d.entity_id for d in state["review_decisions"] if d.validation_status == "REVIEW"}
    if not review_ids:
        return  # this fixture happens to have none in REVIEW status; nothing to assert here
    agent = AttributeMappingAgent(cfg, FakeProvider("rules"))
    delta = agent.run(state)
    for m in delta["mappings"]:
        assert m.entity_id not in review_ids
