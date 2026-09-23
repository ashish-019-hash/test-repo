"""Unit tests for AttributeStorageAgent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from doc_extractor.agents.attribute_storage import AttributeStorageAgent
from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import StageValidationError
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.schemas.attribute import Attribute, EntityBinding, ScoreBreakdown
from doc_extractor.schemas.common import Provenance
from doc_extractor.schemas.state import PipelineState, StageName, StageRecord


@dataclass
class _FakeProvider:
    name: str = "rules"


def _agent(cfg: AppConfig) -> AttributeStorageAgent:
    return AttributeStorageAgent(cfg, _FakeProvider())


def _make_attribute(n: int = 0) -> Attribute:
    return Attribute(
        attribute_id=f"attr-test{n:04d}",
        attribute_name=f"cir{n}",
        display_name=f"CIR{n}",
        attribute_type="integer",
        entity="EVC",
        binding=EntityBinding(entity_name="EVC", scope="table_subject", confidence=0.9, evidence="EVC"),
        source_document="doc-test",
        source_chunk="chunk-doc-test-p1-0000",
        source_text="| CIR | Integer | Mbps | M | 10-1000 | 100 | Committed Information Rate |",
        provenance=Provenance(file="telecom_spec.md", page=1, block_id="blk-doc-test-p1-0000"),
        confidence=0.9,
        score=ScoreBreakdown(hits=[], raw_sum=0.9, clamped=0.9, route="accept"),
        origin="spec_table",
    )


def _state_with_out_dir(tmp_path: Path, attributes: list[Attribute]) -> PipelineState:
    return {
        "attributes": attributes,
        "out_dir": str(tmp_path),
        "stages": {
            "attribute_extraction": StageRecord(stage="attribute_extraction", status="succeeded", attempts=1)
        },
    }


def test_execute_writes_attributes_json(cfg: AppConfig, tmp_path: Path) -> None:
    agent = _agent(cfg)
    attributes = [_make_attribute(0), _make_attribute(1)]
    state = _state_with_out_dir(tmp_path, attributes)
    delta = agent.execute(state, TraceCollector(stage=StageName.attribute_storage))
    path = Path(delta["attributes_path"])
    assert path.is_file()
    assert path == tmp_path / "attributes.json"


def test_run_round_trips_without_change(cfg: AppConfig, tmp_path: Path) -> None:
    agent = _agent(cfg)
    attributes = [_make_attribute(0)]
    state = _state_with_out_dir(tmp_path, attributes)
    result = agent.run(state)
    assert result["stages"]["attribute_storage"].status == "succeeded"
    assert Path(result["attributes_path"]).is_file()


def test_empty_attribute_list_is_valid_input(cfg: AppConfig, tmp_path: Path) -> None:
    agent = _agent(cfg)
    state = _state_with_out_dir(tmp_path, [])
    agent.validate_input(state)  # must not raise
    result = agent.run(state)
    assert result["stages"]["attribute_storage"].status == "succeeded"
    written = Path(result["attributes_path"]).read_text(encoding="utf-8")
    assert written.strip() == "[]"


def test_validate_input_rejects_missing_key(cfg: AppConfig, tmp_path: Path) -> None:
    agent = _agent(cfg)
    state: PipelineState = {
        "out_dir": str(tmp_path),
        "stages": {
            "attribute_extraction": StageRecord(stage="attribute_extraction", status="succeeded", attempts=1)
        },
    }
    with pytest.raises(StageValidationError):
        agent.validate_input(state)


def test_validate_input_rejects_failed_predecessor(cfg: AppConfig, tmp_path: Path) -> None:
    agent = _agent(cfg)
    state: PipelineState = {
        "attributes": [],
        "out_dir": str(tmp_path),
        "stages": {
            "attribute_extraction": StageRecord(stage="attribute_extraction", status="failed", attempts=3)
        },
    }
    with pytest.raises(StageValidationError):
        agent.validate_input(state)


def test_validate_output_detects_tampered_file(cfg: AppConfig, tmp_path: Path) -> None:
    agent = _agent(cfg)
    attributes = [_make_attribute(0)]
    state = _state_with_out_dir(tmp_path, attributes)
    delta = agent.execute(state, TraceCollector(stage=StageName.attribute_storage))
    path = Path(delta["attributes_path"])
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(StageValidationError):
        agent.validate_output(delta, state)
