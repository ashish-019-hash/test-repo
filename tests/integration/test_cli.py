"""`doc-extractor` CLI: subcommands and exit codes (run through a real subprocess)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from doc_extractor.cli import EXIT_OK, EXIT_STAGE_FAILED, EXIT_USAGE, main
from doc_extractor.observability import configure_logging
from doc_extractor.schemas.state import STAGE_ORDER

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """In-process `main()` calls reconfigure structlog; put the defaults back afterwards."""
    yield
    configure_logging()


def _cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if not k.startswith("AZURE_OPENAI_")}
    base["PYTHONPATH"] = str(REPO / "src")
    base.pop("LLM_PROVIDER", None)
    base.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "doc_extractor.cli", *args],
        capture_output=True,
        text=True,
        env=base,
        cwd=REPO,
        check=False,
    )


def test_stages_lists_all_nine_in_order() -> None:
    proc = _cli("stages")
    assert proc.returncode == EXIT_OK
    lines = [ln for ln in proc.stdout.splitlines() if ln[:3].strip().rstrip(".").isdigit()]
    assert [ln.split()[1] for ln in lines] == [str(s) for s in STAGE_ORDER]
    assert "requires:" in proc.stdout and "produces:" in proc.stdout

    proc = _cli("stages", "--json")
    rows = json.loads(proc.stdout)
    assert [r["stage"] for r in rows] == [str(s) for s in STAGE_ORDER]
    assert rows[0]["requires"] == ["input_path"] and rows[-1]["produces"] == ["final_output"]


def test_config_show_prints_effective_config_and_hash(tmp_path: Path) -> None:
    proc = _cli("config", "--show", "--provider", "rules")
    assert proc.returncode == EXIT_OK, proc.stderr
    data = json.loads(proc.stdout)
    assert data["llm"]["provider"] == "rules"
    assert len(data["config_hash"]) == 16
    assert data["scoring"]["routing"]["accept"] == 0.85

    override = tmp_path / "cfg.yaml"
    override.write_text("scoring:\n  routing:\n    accept: 0.9\n", encoding="utf-8")
    proc2 = _cli("config", "--show", "--provider", "rules", "--config", str(override))
    data2 = json.loads(proc2.stdout)
    assert data2["scoring"]["routing"]["accept"] == 0.9
    assert data2["config_hash"] != data["config_hash"]


def test_run_falls_back_to_rules_and_writes_outputs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    proc = _cli(
        "run",
        str(REPO / "tests/fixtures/telecom_spec.md"),
        "--out",
        str(out),
        "--graph",
        "json",
        "--log-format",
        "json",
        "--log-level",
        "WARNING",
    )
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "provider=rules" in proc.stdout
    assert "llm.fallback" in proc.stderr  # auto mode logged the fallback (no Azure env)
    for name in (
        "final.json",
        "entities.json",
        "mappings.json",
        "attributes.json",
        "discarded.json",
        "review_queue.json",
        "run.json",
        "graph.json",
        "checkpoints.sqlite",
    ):
        assert (out / name).exists(), name
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["provider"] == "rules" and run["status"] == "succeeded"


def test_run_uses_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "LLM_PROVIDER=rules\nDOC_EXTRACTOR__scoring__routing__accept=0.95\n", encoding="utf-8"
    )
    proc = _cli("config", "--show", "--env-file", str(env_file))
    assert proc.returncode == EXIT_OK, proc.stderr
    data = json.loads(proc.stdout)
    assert data["llm"]["provider"] == "rules"
    assert data["scoring"]["routing"]["accept"] == 0.95


def test_usage_errors_exit_3(tmp_path: Path) -> None:
    proc = _cli("run", str(tmp_path / "missing.pdf"), "--out", str(tmp_path / "out"), "--provider", "rules")
    assert proc.returncode == EXIT_USAGE
    assert "Input file not found" in proc.stderr

    proc = _cli(
        "run",
        str(REPO / "tests/fixtures/telecom_spec.md"),
        "--out",
        str(tmp_path / "out2"),
        "--provider",
        "rules",
        "--resume",
    )
    assert proc.returncode == EXIT_USAGE
    assert "Nothing to resume" in proc.stderr

    proc = _cli(
        "run",
        str(REPO / "tests/fixtures/telecom_spec.md"),
        "--out",
        str(tmp_path / "o"),
        "--provider",
        "azure",
    )
    assert proc.returncode == EXIT_USAGE
    assert "AZURE_OPENAI_ENDPOINT" in proc.stderr

    proc = _cli("config", "--show", "--config", str(tmp_path / "nope.yaml"))
    assert proc.returncode == EXIT_USAGE

    proc = _cli("run", str(REPO / "tests/fixtures/telecom_spec.md"), "--resume-from", "bogus")
    assert proc.returncode == EXIT_USAGE  # argparse usage errors are mapped to 3, not 2
    assert "invalid choice" in proc.stderr


def test_stage_failure_exits_2_with_resume_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixtures_dir: Path
) -> None:
    """In-process: patch the reviewer to fail, run `main`, check the exit code and message."""
    from doc_extractor.agents import entity_reviewer
    from doc_extractor.exceptions import LLMTransientError

    def boom(self, state, trace):  # type: ignore[no-untyped-def]
        raise LLMTransientError("boom")

    monkeypatch.setattr(entity_reviewer.EntityReviewerAgent, "execute", boom)
    monkeypatch.setenv("LLM_PROVIDER", "rules")
    monkeypatch.setenv("DOC_EXTRACTOR__retry__max_attempts", "1")
    monkeypatch.setenv("DOC_EXTRACTOR__retry__backoff_seconds", "0")
    out = tmp_path / "out"
    code = main(["run", str(fixtures_dir / "telecom_spec.md"), "--out", str(out), "--log-level", "ERROR"])
    assert code == EXIT_STAGE_FAILED
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["stages"]["entity_reviewer"]["status"] == "failed"


def test_stage_failure_message_names_resume_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixtures_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from doc_extractor.agents import entity_reviewer
    from doc_extractor.exceptions import LLMTransientError

    def boom(self, state, trace):  # type: ignore[no-untyped-def]
        raise LLMTransientError("boom")

    monkeypatch.setattr(entity_reviewer.EntityReviewerAgent, "execute", boom)
    monkeypatch.setenv("LLM_PROVIDER", "rules")
    monkeypatch.setenv("DOC_EXTRACTOR__retry__max_attempts", "1")
    out = tmp_path / "out"
    main(["run", str(fixtures_dir / "telecom_spec.md"), "--out", str(out), "--log-level", "ERROR"])
    err = capsys.readouterr().err
    assert "Stage 'entity_reviewer' failed" in err
    assert "--resume" in err and "--resume-from entity_reviewer" in err
