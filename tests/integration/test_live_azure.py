"""Live smoke test against Azure OpenAI.

Runs only with `pytest -m live` AND when all five `AZURE_OPENAI_*` variables are set
(a `.env` in the repository root is loaded automatically). Asserts schema validity and
that every evidence string is a literal substring of its chunk; it does not compare
against the golden file because model output is only best-effort deterministic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from doc_extractor.config import load_config
from doc_extractor.graph.runner import run_pipeline
from doc_extractor.llm.settings import ENV_VARS, load_dotenv_file
from doc_extractor.schemas.output import FinalOutput

pytestmark = pytest.mark.live

load_dotenv_file()
_MISSING = [name for name in ENV_VARS if not os.environ.get(name)]


@pytest.mark.skipif(bool(_MISSING), reason=f"Azure OpenAI env vars not set: {', '.join(_MISSING)}")
def test_live_azure_run(fixtures_dir: Path, tmp_path: Path) -> None:
    cfg = load_config(env={**os.environ, "LLM_PROVIDER": "azure"})
    out = tmp_path / "out"
    result = run_pipeline(fixtures_dir / "telecom_spec.md", out, cfg)
    assert result.ok, result.error
    assert result.metadata.provider == "azure"
    assert result.metadata.model

    final = FinalOutput.model_validate(json.loads((out / "final.json").read_text(encoding="utf-8")))
    assert final.attributes and final.entities
    chunks = {c.chunk_id: c for c in result.state["chunks"]}
    for attr in final.attributes:
        for ev in attr.evidence:
            assert ev.text in chunks[ev.chunk_id].source_text
    assert (out / "llm_cache").is_dir() and any((out / "llm_cache").iterdir())
