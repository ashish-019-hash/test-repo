"""Shared helpers for the Group D integration tests (graph, runner, CLI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from doc_extractor.config.models import AppConfig
from doc_extractor.graph.runner import RunResult, run_pipeline

FORMATS = ("md", "pdf", "docx")
EXPECTED_DIR = Path(__file__).resolve().parent / "fixtures" / "expected"

DETERMINISTIC_FILES = (
    "attributes.json",
    "discarded.json",
    "entities.json",
    "mappings.json",
    "final.json",
    "review_queue.json",
)


def run_fixture(fixtures_dir: Path, fmt: str, out_dir: Path, cfg: AppConfig, **kwargs: Any) -> RunResult:
    """Run the pipeline on `telecom_spec.<fmt>` with an empty environment (no Azure)."""
    return run_pipeline(fixtures_dir / f"telecom_spec.{fmt}", out_dir, cfg, env={}, **kwargs)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
