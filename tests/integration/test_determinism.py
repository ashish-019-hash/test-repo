"""Two independent runs must produce byte-identical deterministic outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_extractor.config.models import AppConfig
from helpers_d import DETERMINISTIC_FILES, FORMATS, read_json, run_fixture


@pytest.mark.parametrize("fmt", FORMATS)
def test_two_runs_are_byte_identical(
    fmt: str, fixtures_dir: Path, tmp_path: Path, rules_cfg: AppConfig
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    ra = run_fixture(fixtures_dir, fmt, a, rules_cfg)
    rb = run_fixture(fixtures_dir, fmt, b, rules_cfg)
    assert ra.ok and rb.ok
    for name in DETERMINISTIC_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), f"{name} differs between runs"
    # graph export, when enabled, is deterministic too
    assert ra.metadata.document_id == rb.metadata.document_id
    assert ra.thread_id == rb.thread_id


def test_graph_export_is_deterministic(fixtures_dir: Path, tmp_path: Path, rules_cfg: AppConfig) -> None:
    cfg = rules_cfg.model_copy(
        update={"output": rules_cfg.output.model_copy(update={"graph_export": "json"})}
    )
    a, b = tmp_path / "a", tmp_path / "b"
    assert run_fixture(fixtures_dir, "md", a, cfg).ok
    assert run_fixture(fixtures_dir, "md", b, cfg).ok
    assert (a / "graph.json").read_bytes() == (b / "graph.json").read_bytes()
    g = read_json(a / "graph.json")
    assert g["nodes"] and g["edges"]


def test_run_json_holds_the_volatile_fields(fixtures_dir: Path, tmp_path: Path, rules_cfg: AppConfig) -> None:
    out = tmp_path / "out"
    result = run_fixture(fixtures_dir, "md", out, rules_cfg)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "succeeded"
    assert run["run_id"] == result.metadata.run_id
    assert run["provider"] == "rules"
    assert run["config_hash"] == rules_cfg.config_hash
    assert run["thread_id"] == result.thread_id
    assert run["input_file"] == "telecom_spec.md"
    assert run["started_at"] and run["finished_at"]
    assert len(run["traces"]) == 9
    assert all(rec["duration_ms"] is not None for rec in run["stages"].values())
