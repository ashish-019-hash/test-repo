"""Unit tests for per-stage pipeline state snapshots."""

from __future__ import annotations

from pathlib import Path

from doc_extractor.chunking import chunk_document
from doc_extractor.config.models import AppConfig
from doc_extractor.ingest import load_document
from doc_extractor.schemas.state import PipelineState, StageRecord
from doc_extractor.storage.snapshots import load_snapshot, save_snapshot, snapshot_path


def _make_state(cfg: AppConfig, telecom_spec_md: Path) -> PipelineState:
    document = load_document(telecom_spec_md, cfg)
    chunks = chunk_document(document, cfg)
    state: PipelineState = {
        "input_path": str(telecom_spec_md),
        "out_dir": "/tmp/does-not-matter",
        "config_hash": cfg.config_hash,
        "document": document,
        "chunks": chunks,
        "stages": {
            "document_processing": StageRecord(stage="document_processing", status="succeeded", attempts=1),
            "chunking": StageRecord(stage="chunking", status="succeeded", attempts=1),
        },
        "traces": [],
        "human_review_queue": [],
    }
    return state


def test_save_snapshot_writes_expected_path(tmp_path: Path, cfg: AppConfig, telecom_spec_md: Path) -> None:
    state = _make_state(cfg, telecom_spec_md)
    path = save_snapshot(tmp_path, 2, "chunking", state)
    assert path == snapshot_path(tmp_path, 2, "chunking")
    assert path == tmp_path / "stages" / "02_chunking.json"
    assert path.is_file()


def test_save_and_load_snapshot_round_trips_document_and_chunks(
    tmp_path: Path, cfg: AppConfig, telecom_spec_md: Path
) -> None:
    state = _make_state(cfg, telecom_spec_md)
    path = save_snapshot(tmp_path, 2, "chunking", state)
    loaded = load_snapshot(path)

    assert loaded["document"].document_id == state["document"].document_id
    assert loaded["document"].blocks[0].text == state["document"].blocks[0].text
    assert len(loaded["chunks"]) == len(state["chunks"])
    for original, reloaded in zip(state["chunks"], loaded["chunks"], strict=True):
        assert reloaded.chunk_id == original.chunk_id
        assert reloaded.source_text == original.source_text
        assert reloaded.heading_path == original.heading_path

    assert loaded["input_path"] == state["input_path"]
    assert loaded["config_hash"] == state["config_hash"]
    assert set(loaded["stages"]) == {"document_processing", "chunking"}
    assert loaded["stages"]["chunking"].status == "succeeded"
    assert isinstance(loaded["stages"]["chunking"], StageRecord)


def test_load_snapshot_from_path_string(tmp_path: Path, cfg: AppConfig, telecom_spec_md: Path) -> None:
    state = _make_state(cfg, telecom_spec_md)
    path = save_snapshot(tmp_path, 1, "document_processing", state)
    loaded = load_snapshot(str(path))
    assert loaded["document"].document_id == state["document"].document_id
