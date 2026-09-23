"""Unit tests for ChunkingAgent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from doc_extractor.agents.chunking import ChunkingAgent
from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import StageValidationError
from doc_extractor.ingest import load_document
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.schemas.state import PipelineState, StageName, StageRecord


@dataclass
class _FakeProvider:
    name: str = "rules"


def _agent(cfg: AppConfig) -> ChunkingAgent:
    return ChunkingAgent(cfg, _FakeProvider())


def _state(cfg: AppConfig, telecom_spec_md: Path) -> PipelineState:
    document = load_document(telecom_spec_md, cfg)
    return {
        "document": document,
        "stages": {
            "document_processing": StageRecord(stage="document_processing", status="succeeded", attempts=1)
        },
    }


def test_execute_produces_chunks(cfg: AppConfig, telecom_spec_md: Path) -> None:
    agent = _agent(cfg)
    state = _state(cfg, telecom_spec_md)
    delta = agent.execute(state, TraceCollector(stage=StageName.chunking))
    assert set(delta) == {"chunks"}
    assert delta["chunks"]


def test_run_produces_stage_record(cfg: AppConfig, telecom_spec_md: Path) -> None:
    agent = _agent(cfg)
    state = _state(cfg, telecom_spec_md)
    result = agent.run(state)
    assert result["stages"]["chunking"].status == "succeeded"
    assert result["chunks"]


def test_validate_input_requires_document(cfg: AppConfig) -> None:
    agent = _agent(cfg)
    with pytest.raises(StageValidationError):
        agent.validate_input({})


def test_validate_output_catches_broken_source_text(cfg: AppConfig, telecom_spec_md: Path) -> None:
    agent = _agent(cfg)
    state = _state(cfg, telecom_spec_md)
    delta = agent.execute(state, TraceCollector(stage=StageName.chunking))
    broken = list(delta["chunks"])
    broken[0] = broken[0].model_copy(update={"source_text": "tampered"})
    with pytest.raises(StageValidationError):
        agent.validate_output({"chunks": broken}, state)


def test_validate_output_catches_uncovered_blocks(cfg: AppConfig, telecom_spec_md: Path) -> None:
    agent = _agent(cfg)
    state = _state(cfg, telecom_spec_md)
    delta = agent.execute(state, TraceCollector(stage=StageName.chunking))
    with pytest.raises(StageValidationError):
        agent.validate_output({"chunks": delta["chunks"][:-1]}, state)


def test_validate_output_catches_duplicate_ids(cfg: AppConfig, telecom_spec_md: Path) -> None:
    agent = _agent(cfg)
    state = _state(cfg, telecom_spec_md)
    delta = agent.execute(state, TraceCollector(stage=StageName.chunking))
    duped = list(delta["chunks"]) + [delta["chunks"][0]]
    with pytest.raises(StageValidationError):
        agent.validate_output({"chunks": duped}, state)
