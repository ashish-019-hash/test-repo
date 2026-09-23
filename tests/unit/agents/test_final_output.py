"""Tests for Agent 9: FinalOutputAgent."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from doc_extractor.agents.attribute_mapping import AttributeMappingAgent
from doc_extractor.agents.entity_generation import EntityGenerationAgent
from doc_extractor.agents.entity_normalization import EntityNormalizationAgent
from doc_extractor.agents.entity_reviewer import EntityReviewerAgent
from doc_extractor.agents.final_output import FinalOutputAgent
from doc_extractor.config import load_config
from doc_extractor.schemas.state import StageName, StageRecord
from helpers_c import FakeProvider, base_state


def _state_through_mapping(cfg):
    provider = FakeProvider("rules")
    state = base_state(StageName.attribute_storage)
    for stage, agent_cls in [
        (StageName.entity_generation, EntityGenerationAgent),
        (StageName.entity_normalization, EntityNormalizationAgent),
        (StageName.entity_reviewer, EntityReviewerAgent),
        (StageName.attribute_mapping, AttributeMappingAgent),
    ]:
        agent = agent_cls(cfg, provider)
        state.update(agent.run(state))
        state["stages"][str(stage)] = StageRecord(stage=stage, attempts=1, status="succeeded")
    return state


def test_final_output_writes_expected_files(cfg) -> None:
    state = _state_through_mapping(cfg)
    with tempfile.TemporaryDirectory() as d:
        state["out_dir"] = d
        agent = FinalOutputAgent(cfg, FakeProvider("rules"))
        delta = agent.run(state)
        final = delta["final_output"]
        written = sorted(p.name for p in Path(d).iterdir())
        assert written == ["entities.json", "final.json", "mappings.json", "review_queue.json"]
        assert final.document.document_id == state["document"].document_id
        raw = json.loads((Path(d) / "final.json").read_text())
        assert raw["schema_version"] == "1.0"


def test_dangling_and_unmapped_attributes_are_reported(cfg) -> None:
    state = _state_through_mapping(cfg)
    with tempfile.TemporaryDirectory() as d:
        state["out_dir"] = d
        agent = FinalOutputAgent(cfg, FakeProvider("rules"))
        final = agent.run(state)["final_output"]
        assert final.dangling_attribute_ids == ["attr-provisioningtime"]
        assert "attr-provisioningtime" in final.unmapped_attribute_ids


def test_trace_covers_all_six_edge_kinds(cfg) -> None:
    state = _state_through_mapping(cfg)
    with tempfile.TemporaryDirectory() as d:
        state["out_dir"] = d
        agent = FinalOutputAgent(cfg, FakeProvider("rules"))
        final = agent.run(state)["final_output"]
        kinds = {e.kind for e in final.trace}
        assert kinds == {
            "document>chunk",
            "chunk>attribute",
            "attribute>entity",
            "entity>canonical",
            "attribute>mapping",
            "mapping>entity",
        }


def test_review_section_routes_reject_and_review_decisions(cfg) -> None:
    state = _state_through_mapping(cfg)
    with tempfile.TemporaryDirectory() as d:
        state["out_dir"] = d
        agent = FinalOutputAgent(cfg, FakeProvider("rules"))
        final = agent.run(state)["final_output"]
        statuses = {d.entity_id: d.validation_status for d in state["review_decisions"]}
        review_ids = {eid for eid, s in statuses.items() if s == "REVIEW"}
        reject_ids = {eid for eid, s in statuses.items() if s == "REJECT"}
        assert {d.entity_id for d in final.review.entities_requiring_review} == review_ids
        assert {d.entity_id for d in final.review.rejected_entities} == reject_ids


def test_graph_export_writes_graph_json_when_configured() -> None:
    cfg = load_config(env={"LLM_PROVIDER": "rules", "DOC_EXTRACTOR__output__graph_export": "json"})
    state = _state_through_mapping(cfg)
    with tempfile.TemporaryDirectory() as d:
        state["out_dir"] = d
        agent = FinalOutputAgent(cfg, FakeProvider("rules"))
        agent.run(state)
        assert (Path(d) / "graph.json").is_file()
        payload = json.loads((Path(d) / "graph.json").read_text())
        assert payload["directed"] is True
        assert payload["nodes"]


def test_no_graph_file_when_not_configured(cfg) -> None:
    state = _state_through_mapping(cfg)
    with tempfile.TemporaryDirectory() as d:
        state["out_dir"] = d
        agent = FinalOutputAgent(cfg, FakeProvider("rules"))
        agent.run(state)
        assert not (Path(d) / "graph.json").exists()
        assert not (Path(d) / "graph.graphml").exists()


def test_final_output_is_stable_under_canonical_json_round_trip(cfg) -> None:
    from doc_extractor.storage.canonical_json import dumps

    state = _state_through_mapping(cfg)
    with tempfile.TemporaryDirectory() as d:
        state["out_dir"] = d
        agent = FinalOutputAgent(cfg, FakeProvider("rules"))
        final = agent.run(state)["final_output"]
        from doc_extractor.schemas.output import FinalOutput

        reloaded = FinalOutput.model_validate(json.loads(dumps(final)))
        assert dumps(reloaded) == dumps(final)
