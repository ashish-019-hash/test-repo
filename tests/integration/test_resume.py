"""Failure, `--resume` and `--resume-from` behaviour of the runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from doc_extractor.agents import AGENT_REGISTRY
from doc_extractor.agents.base import BaseAgent
from doc_extractor.agents.entity_reviewer import EntityReviewerAgent
from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import ConfigError, IngestError, LLMTransientError
from doc_extractor.graph.runner import run_pipeline, strip_from_stage
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.schemas.state import STAGE_ORDER, PipelineState, StageName
from doc_extractor.storage.snapshots import load_snapshot
from helpers_d import DETERMINISTIC_FILES, run_fixture

CALLS: dict[str, int] = {}


def _counting(cls: type[BaseAgent]) -> type[BaseAgent]:
    """Subclass `cls` so every `execute` call is counted under the stage name."""

    class Counting(cls):  # type: ignore[valid-type,misc]
        def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
            CALLS[str(self.name)] = CALLS.get(str(self.name), 0) + 1
            return super().execute(state, trace)

    Counting.__name__ = f"Counting{cls.__name__}"
    return Counting


class FailingReviewer(EntityReviewerAgent):
    """Raises a transient error on every attempt so the stage exhausts its retries."""

    fail: ClassVar[bool] = True

    def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
        CALLS["entity_reviewer"] = CALLS.get("entity_reviewer", 0) + 1
        if FailingReviewer.fail:
            raise LLMTransientError("simulated provider outage")
        return super().execute(state, trace)


@pytest.fixture
def fast_cfg(rules_cfg: AppConfig) -> AppConfig:
    return rules_cfg.model_copy(
        update={"retry": rules_cfg.retry.model_copy(update={"max_attempts": 2, "backoff_seconds": 0.0})}
    )


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()
    FailingReviewer.fail = True


def _overrides() -> dict[StageName, type[BaseAgent]]:
    ov = {stage: _counting(cls) for stage, cls in AGENT_REGISTRY.items()}
    ov[StageName.entity_reviewer] = FailingReviewer
    return ov


def test_failure_then_resume(fixtures_dir: Path, tmp_path: Path, fast_cfg: AppConfig) -> None:
    out = tmp_path / "out"
    overrides = _overrides()

    failed = run_fixture(fixtures_dir, "md", out, fast_cfg, agent_overrides=overrides)
    assert not failed.ok
    assert failed.failed_stage == "entity_reviewer"
    assert CALLS["entity_reviewer"] == 2  # max_attempts
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["stages"]["entity_reviewer"]["status"] == "failed"
    assert run["stages"]["entity_reviewer"]["attempts"] == 2
    assert "simulated provider outage" in run["stages"]["entity_reviewer"]["error"]
    for stage in STAGE_ORDER[:6]:
        assert run["stages"][str(stage)]["status"] == "succeeded"
    for stage in STAGE_ORDER[7:]:
        assert run["stages"][str(stage)]["status"] == "pending"
    assert (out / "stages" / "06_entity_normalization.json").is_file()
    assert not (out / "stages" / "07_entity_reviewer.json").exists()
    assert not (out / "final.json").exists()
    assert (out / "checkpoints.sqlite").exists()

    # heal the provider and resume: earlier stages are NOT re-executed
    FailingReviewer.fail = False
    before = dict(CALLS)
    resumed = run_fixture(fixtures_dir, "md", out, fast_cfg, agent_overrides=overrides, resume=True)
    assert resumed.ok, resumed.error
    for stage in STAGE_ORDER[:6]:
        assert CALLS[str(stage)] == before[str(stage)] == 1, stage
    assert CALLS["entity_reviewer"] == 3
    assert CALLS["attribute_mapping"] == 1 and CALLS["final_output"] == 1
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "succeeded"
    assert all(rec["status"] == "succeeded" for rec in run["stages"].values())
    assert run["stages"]["entity_reviewer"]["attempts"] == 1
    for name in DETERMINISTIC_FILES:
        assert (out / name).is_file(), name

    # the healed output equals a clean run
    clean = tmp_path / "clean"
    assert run_fixture(fixtures_dir, "md", clean, fast_cfg).ok
    assert (out / "final.json").read_bytes() == (clean / "final.json").read_bytes()


def test_resume_from_reruns_only_later_stages(
    fixtures_dir: Path, tmp_path: Path, fast_cfg: AppConfig
) -> None:
    out = tmp_path / "out"
    overrides = {stage: _counting(cls) for stage, cls in AGENT_REGISTRY.items()}
    assert run_fixture(fixtures_dir, "md", out, fast_cfg, agent_overrides=overrides).ok
    baseline = (out / "final.json").read_bytes()
    CALLS.clear()

    result = run_fixture(
        fixtures_dir, "md", out, fast_cfg, agent_overrides=overrides, resume_from="entity_generation"
    )
    assert result.ok, result.error
    assert set(CALLS) == {
        "entity_generation",
        "entity_normalization",
        "entity_reviewer",
        "attribute_mapping",
        "final_output",
    }
    assert all(n == 1 for n in CALLS.values())
    assert (out / "final.json").read_bytes() == baseline
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert all(rec["status"] == "succeeded" for rec in run["stages"].values())
    # traces are not duplicated by the re-seeded thread
    assert len(run["traces"]) == 9
    assert [t["stage"] for t in run["traces"]] == [str(s) for s in STAGE_ORDER]


def test_resume_from_first_stage_is_a_fresh_run(
    fixtures_dir: Path, tmp_path: Path, fast_cfg: AppConfig
) -> None:
    out = tmp_path / "out"
    overrides = {stage: _counting(cls) for stage, cls in AGENT_REGISTRY.items()}
    assert run_fixture(fixtures_dir, "md", out, fast_cfg, agent_overrides=overrides).ok
    CALLS.clear()
    assert run_fixture(
        fixtures_dir, "md", out, fast_cfg, agent_overrides=overrides, resume_from="document_processing"
    ).ok
    assert all(CALLS[str(s)] == 1 for s in STAGE_ORDER)


def test_resume_from_accepts_stage_enum(fixtures_dir: Path, tmp_path: Path, fast_cfg: AppConfig) -> None:
    out = tmp_path / "out"
    assert run_fixture(fixtures_dir, "md", out, fast_cfg).ok
    assert run_fixture(fixtures_dir, "md", out, fast_cfg, resume_from=StageName.final_output).ok


def test_strip_from_stage_removes_later_outputs(
    fixtures_dir: Path, tmp_path: Path, fast_cfg: AppConfig
) -> None:
    out = tmp_path / "out"
    assert run_fixture(fixtures_dir, "md", out, fast_cfg).ok
    full = load_snapshot(out / "stages" / "09_final_output.json")
    stripped = strip_from_stage(full, StageName.entity_reviewer, tmp_path / "elsewhere")
    assert "entities" in stripped and "normalized_entities" in stripped
    for key in ("duplicate_groups", "canonical_entities", "review_decisions", "mappings", "final_output"):
        assert key not in stripped
    assert set(stripped["stages"]) == {str(s) for s in STAGE_ORDER[:6]}
    assert {t.stage for t in stripped["traces"]} == set(STAGE_ORDER[:6])
    assert all(h.kind == "attribute_review" for h in stripped["human_review_queue"])
    assert stripped["out_dir"] == str(tmp_path / "elsewhere")
    assert stripped["resume_from"] == "entity_reviewer"


def test_invalid_invocations(fixtures_dir: Path, tmp_path: Path, fast_cfg: AppConfig) -> None:
    out = tmp_path / "out"
    with pytest.raises(IngestError):
        run_pipeline(tmp_path / "missing.md", out, fast_cfg, env={})
    with pytest.raises(ConfigError, match="Nothing to resume"):
        run_fixture(fixtures_dir, "md", out, fast_cfg, resume=True)
    with pytest.raises(ConfigError, match="snapshot .* not found"):
        run_fixture(fixtures_dir, "md", out, fast_cfg, resume_from="entity_generation")
    with pytest.raises(ConfigError, match="Unknown stage"):
        run_fixture(fixtures_dir, "md", out, fast_cfg, resume_from="not_a_stage")
    with pytest.raises(ConfigError, match="mutually exclusive"):
        run_fixture(fixtures_dir, "md", out, fast_cfg, resume=True, resume_from="chunking")
