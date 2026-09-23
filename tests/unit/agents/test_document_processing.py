"""Unit tests for DocumentProcessingAgent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from doc_extractor.agents.document_processing import DocumentProcessingAgent
from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import StageValidationError
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.schemas.state import PipelineState, StageName


@dataclass
class _FakeProvider:
    name: str = "rules"


def _agent(cfg: AppConfig) -> DocumentProcessingAgent:
    return DocumentProcessingAgent(cfg, _FakeProvider())


def test_execute_loads_document(cfg: AppConfig, telecom_spec_md: Path) -> None:
    agent = _agent(cfg)
    state: PipelineState = {"input_path": str(telecom_spec_md)}
    trace = TraceCollector(stage=StageName.document_processing)
    delta = agent.execute(state, trace)
    assert set(delta) == {"document"}
    document = delta["document"]
    assert document.format == "md"
    assert document.blocks


def test_run_produces_stage_record_and_document(cfg: AppConfig, telecom_spec_md: Path) -> None:
    agent = _agent(cfg)
    state: PipelineState = {"input_path": str(telecom_spec_md)}
    result = agent.run(state)
    assert result["stages"]["document_processing"].status == "succeeded"
    assert result["document"].document_id.startswith("doc-")


def test_validate_input_rejects_missing_input_path(cfg: AppConfig) -> None:
    agent = _agent(cfg)
    with pytest.raises(StageValidationError):
        agent.validate_input({})


def test_validate_output_rejects_document_with_no_blocks(cfg: AppConfig, telecom_spec_md: Path) -> None:
    agent = _agent(cfg)
    state: PipelineState = {"input_path": str(telecom_spec_md)}
    document = agent.execute(state, TraceCollector(stage=StageName.document_processing))["document"]
    empty_document = document.model_copy(update={"blocks": []})
    with pytest.raises(StageValidationError):
        agent.validate_output({"document": empty_document}, state)
