"""Pipeline runner: fresh runs, `--resume`, `--resume-from <stage>`, `run.json`, snapshots.

Recovery model
--------------
* Fresh run: the thread (document id + config hash) is cleared and the graph is
  invoked with the initial input. After every successful node a full state snapshot
  is written to ``<out_dir>/stages/NN_<stage>.json``.
* ``resume=True``: the graph is invoked with ``None`` on the same thread; LangGraph
  continues after the last checkpointed node (the failed node re-runs).
* ``resume_from=<stage>``: the snapshot of the predecessor stage is loaded, later
  stage outputs and records are stripped, the thread is cleared and re-seeded with
  ``update_state(as_node=<predecessor>)``, then the graph continues from ``<stage>``.
* A stage that exhausts its retries raises ``StageFailedError``; the runner records
  the failure in ``run.json`` (status ``failed``) and returns instead of raising, so
  callers can map it to an exit code. The checkpoint stays on disk.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from doc_extractor import __version__
from doc_extractor.agents import AGENT_REGISTRY
from doc_extractor.agents.base import BaseAgent
from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import ConfigError, IngestError, StageFailedError
from doc_extractor.graph.build import (
    build_agents,
    compile_graph,
    open_checkpointer,
    thread_config,
    thread_id,
)
from doc_extractor.llm.base import LLMCallResult, LLMProvider, LLMTask
from doc_extractor.llm.factory import build_provider, provider_model_label
from doc_extractor.observability import bind_context, clear_context, get_logger, utc_now_iso
from doc_extractor.schemas.output import RunMetadata
from doc_extractor.schemas.state import STAGE_ORDER, PipelineState, StageName, StageRecord
from doc_extractor.storage import ids
from doc_extractor.storage.snapshots import load_snapshot, save_snapshot, snapshot_path

RUN_FILE = "run.json"
LLM_CACHE_DIR = "llm_cache"

# Which stage emits which kind of human-review item (used to strip items on --resume-from).
_REVIEW_KIND_STAGE: dict[str, StageName] = {
    "attribute_review": StageName.attribute_extraction,
    "possible_duplicate": StageName.entity_reviewer,
    "entity_review": StageName.entity_reviewer,
}


class _FingerprintRecorder:
    """Transparent provider wrapper that remembers the last Azure `system_fingerprint`."""

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.name = inner.name
        self.system_fingerprint: str | None = None

    def call(self, task: LLMTask, payload: dict[str, Any]) -> LLMCallResult:
        result = self.inner.call(task, payload)
        if result.system_fingerprint:
            self.system_fingerprint = result.system_fingerprint
        return result


@dataclass
class RunResult:
    status: str  # "succeeded" | "failed"
    out_dir: Path
    thread_id: str
    metadata: RunMetadata
    state: PipelineState
    failed_stage: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"


def stage_index(stage: StageName) -> int:
    """1-based position used in snapshot file names (`01_document_processing.json`)."""
    return STAGE_ORDER.index(stage) + 1


def _succeeded_count(state: Mapping[str, Any]) -> int:
    return sum(1 for r in (state.get("stages") or {}).values() if r.status == "succeeded")


def _last_succeeded_stage(state: Mapping[str, Any]) -> StageName | None:
    records = state.get("stages") or {}
    last: StageName | None = None
    for stage in STAGE_ORDER:
        rec = records.get(str(stage))
        if rec is not None and rec.status == "succeeded":
            last = stage
    return last


def initial_state(input_path: Path, out_dir: Path, cfg: AppConfig) -> PipelineState:
    return cast(
        PipelineState,
        {
            "input_path": str(input_path),
            "out_dir": str(out_dir),
            "config_hash": cfg.config_hash,
            "resume_from": None,
            "stages": {},
            "traces": [],
            "human_review_queue": [],
        },
    )


def strip_from_stage(state: PipelineState, stage: StageName, out_dir: Path) -> PipelineState:
    """Return a copy of `state` with everything produced at or after `stage` removed.

    Also refreshes `out_dir`/`resume_from` so a snapshot taken in one directory can be
    replayed into another.
    """
    idx = STAGE_ORDER.index(stage)
    later = STAGE_ORDER[idx:]
    drop_keys: set[str] = set()
    for s in later:
        drop_keys.update(AGENT_REGISTRY[s].produces)
    new: dict[str, Any] = {k: v for k, v in dict(state).items() if k not in drop_keys}
    new["stages"] = {k: v for k, v in (state.get("stages") or {}).items() if k not in {str(s) for s in later}}
    keep = set(STAGE_ORDER[:idx])
    new["traces"] = [t for t in state.get("traces", []) if t.stage in keep]
    new["human_review_queue"] = [
        h for h in state.get("human_review_queue", []) if _REVIEW_KIND_STAGE.get(h.kind) in keep
    ]
    new["out_dir"] = str(out_dir)
    new["resume_from"] = str(stage)
    return cast(PipelineState, new)


def _resolve_stage(name: str) -> StageName:
    try:
        return StageName(name)
    except ValueError as exc:
        raise ConfigError(
            f"Unknown stage '{name}'. Valid stages: {', '.join(str(s) for s in STAGE_ORDER)}"
        ) from exc


def _write_run_json(out_dir: Path, meta: RunMetadata) -> Path:
    from doc_extractor.storage.canonical_json import dumps

    path = out_dir / RUN_FILE
    path.write_text(dumps(meta), encoding="utf-8")
    return path


def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path,
    cfg: AppConfig,
    *,
    provider: LLMProvider | None = None,
    env: Mapping[str, str] | None = None,
    resume: bool = False,
    resume_from: str | StageName | None = None,
    agent_overrides: Mapping[StageName, type[BaseAgent]] | None = None,
    log: Any | None = None,
) -> RunResult:
    """Execute (or resume) the pipeline for one document. Never raises `StageFailedError`.

    Raises `ConfigError` / `IngestError` for invalid invocations (missing file, unknown
    stage, nothing to resume, missing snapshot).
    """
    src = Path(input_path)
    out = Path(out_dir)
    if not src.is_file():
        raise IngestError(f"Input file not found: {src}")
    if resume and resume_from is not None:
        raise ConfigError("--resume and --resume-from are mutually exclusive")
    out.mkdir(parents=True, exist_ok=True)

    logger = log or get_logger()
    base_provider = provider or build_provider(cfg, env=env, cache_dir=out / LLM_CACHE_DIR, log=logger)
    model_label = provider_model_label(base_provider)
    recorder = _FingerprintRecorder(base_provider)

    doc_id = ids.document_id(src.read_bytes())
    thread = thread_id(doc_id, cfg.config_hash)
    config = thread_config(thread)
    run_id = uuid.uuid4().hex
    started = utc_now_iso()
    bind_context(run_id=run_id, document_id=doc_id)
    logger.info("run.start", input=src.name, out_dir=str(out), provider=recorder.name, thread_id=thread)

    agents = build_agents(cfg, recorder, log=logger, overrides=agent_overrides)
    failed_stage: str | None = None
    error: str | None = None
    failed_record: StageRecord | None = None

    try:
        with open_checkpointer(out) as saver:
            graph = compile_graph(agents, saver)
            graph_input: Any

            if resume:
                snapshot = graph.get_state(config)
                if not snapshot.values:
                    raise ConfigError(
                        f"Nothing to resume in {out}: no checkpoint for thread {thread}. Run without --resume."
                    )
                graph_input = None
                logger.info("run.resume", after=str(_last_succeeded_stage(snapshot.values)))
            elif resume_from is not None:
                stage = _resolve_stage(str(resume_from))
                saver.delete_thread(thread)
                if stage == STAGE_ORDER[0]:
                    graph_input = initial_state(src, out, cfg)
                else:
                    pred = STAGE_ORDER[STAGE_ORDER.index(stage) - 1]
                    snap_file = snapshot_path(out, stage_index(pred), str(pred))
                    if not snap_file.is_file():
                        raise ConfigError(
                            f"Cannot resume from '{stage}': snapshot {snap_file} not found. "
                            "Run the pipeline at least once first."
                        )
                    seed = strip_from_stage(load_snapshot(snap_file), stage, out)
                    seed["input_path"] = str(src)
                    graph.update_state(config, dict(seed), as_node=str(pred))
                    graph_input = None
                logger.info("run.resume_from", stage=str(stage))
            else:
                saver.delete_thread(thread)
                graph_input = initial_state(src, out, cfg)

            try:
                seen = _succeeded_count(graph.get_state(config).values or {})
                for values in graph.stream(graph_input, config, stream_mode="values"):
                    done = _succeeded_count(values)
                    if done > seen:
                        seen = done
                        last = _last_succeeded_stage(values)
                        if last is not None:
                            save_snapshot(out, stage_index(last), str(last), values)
            except StageFailedError as exc:
                failed_stage = exc.stage
                error = str(exc)
                failed_record = StageRecord(
                    stage=StageName(exc.stage),
                    status="failed",
                    attempts=exc.attempts,
                    error=str(exc.cause),
                    started_at=started,
                    finished_at=utc_now_iso(),
                )
                logger.error("run.failed", stage=exc.stage, attempts=exc.attempts, error=str(exc.cause))

            final_state = cast(PipelineState, dict(graph.get_state(config).values or {}))
    finally:
        clear_context()

    records: dict[str, Any] = {
        k: v.model_dump(mode="json") for k, v in (final_state.get("stages") or {}).items()
    }
    if failed_record is not None:
        records[str(failed_record.stage)] = failed_record.model_dump(mode="json")
    for s in STAGE_ORDER:
        records.setdefault(str(s), StageRecord(stage=s, status="pending").model_dump(mode="json"))
    records = {str(s): records[str(s)] for s in STAGE_ORDER}

    status = "failed" if failed_stage else "succeeded"
    meta = RunMetadata(
        run_id=run_id,
        document_id=doc_id,
        input_file=src.name,
        started_at=started,
        finished_at=utc_now_iso(),
        provider=recorder.name,
        model=model_label,
        system_fingerprint=recorder.system_fingerprint,
        config_hash=cfg.config_hash,
        package_version=__version__,
        status=status,  # type: ignore[arg-type]
        thread_id=thread,
        stages=records,
        traces=[t.model_dump(mode="json") for t in final_state.get("traces", [])],
    )
    _write_run_json(out, meta)
    logger.info("run.done", status=status, out_dir=str(out))
    return RunResult(
        status=status,
        out_dir=out,
        thread_id=thread,
        metadata=meta,
        state=final_state,
        failed_stage=failed_stage,
        error=error,
    )


__all__ = [
    "LLM_CACHE_DIR",
    "RUN_FILE",
    "RunResult",
    "initial_state",
    "run_pipeline",
    "stage_index",
    "strip_from_stage",
]
