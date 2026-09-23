"""BaseAgent: the single contract every stage implements.

Lifecycle of `run(state)`:
  validate_input(state)            -> StageValidationError(phase="input")  (never retried)
  execute(state) -> delta          -> may raise LLMTransientError / StageValidationError(phase="output")
  validate_output(delta, state)    -> StageValidationError(phase="output")
  retry on LLMTransientError and output-validation errors up to cfg.retry.max_attempts
  returns delta + {"stages": {name: StageRecord}, "traces": [TraceRecord, ...]}
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import LLMTransientError, StageFailedError, StageValidationError
from doc_extractor.llm.base import LLMProvider
from doc_extractor.observability.logging import get_logger
from doc_extractor.observability.tracing import TraceCollector, utc_now_iso
from doc_extractor.schemas.state import (
    STAGE_ORDER,
    HumanReviewItem,
    PipelineState,
    StageName,
    StageRecord,
    TraceRecord,
)


class BaseAgent(ABC):
    name: ClassVar[StageName]
    requires: ClassVar[tuple[str, ...]] = ()
    produces: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: AppConfig, provider: LLMProvider, log: Any | None = None) -> None:
        self.cfg = cfg
        self.provider = provider
        self.log = (log or get_logger()).bind(stage=str(self.name))

    # ---- hooks -----------------------------------------------------------------------------
    def validate_input(self, state: PipelineState) -> None:
        """Default: predecessor succeeded (if any) and all `requires` keys are present and non-empty."""
        idx = STAGE_ORDER.index(self.name)
        if idx > 0:
            prev = STAGE_ORDER[idx - 1]
            rec = (state.get("stages") or {}).get(str(prev))
            if rec is None or rec.status != "succeeded":
                raise StageValidationError(
                    str(self.name),
                    f"predecessor stage '{prev}' has status {rec.status if rec else 'missing'}",
                    phase="input",
                )
        for key in self.requires:
            value = state.get(key)
            if value is None or (isinstance(value, list | dict | str) and len(value) == 0):
                raise StageValidationError(
                    str(self.name), f"required state key '{key}' is missing or empty", phase="input"
                )

    @abstractmethod
    def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
        """Do the work. Return a state delta containing only `produces` keys (+ optional human_review_queue)."""

    def validate_output(self, delta: dict[str, Any], state: PipelineState) -> None:
        """Default: every `produces` key is present in the delta and nothing else leaks in."""
        allowed = set(self.produces) | {"human_review_queue"}
        extra = set(delta) - allowed
        if extra:
            raise StageValidationError(
                str(self.name), f"delta contains keys outside `produces`: {sorted(extra)}"
            )
        for key in self.produces:
            if key not in delta:
                raise StageValidationError(str(self.name), f"delta missing produced key '{key}'")
        hri = delta.get("human_review_queue", [])
        if not all(isinstance(i, HumanReviewItem) for i in hri):
            raise StageValidationError(str(self.name), "human_review_queue items must be HumanReviewItem")

    # ---- driver ----------------------------------------------------------------------------
    def run(self, state: PipelineState) -> dict[str, Any]:
        started = utc_now_iso()
        t0 = time.perf_counter()
        self.validate_input(state)  # input errors are never retried
        max_attempts = max(1, self.cfg.retry.max_attempts)
        traces: list[TraceRecord] = []
        last_exc: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            trace = TraceCollector(stage=self.name)
            log = self.log.bind(attempt=attempt)
            log.info("stage.start")
            try:
                delta = self.execute(state, trace)
                self.validate_output(delta, state)
            except (LLMTransientError, StageValidationError) as exc:
                if isinstance(exc, StageValidationError) and exc.phase == "input":
                    raise
                last_exc = exc
                traces.append(trace.finish(event="stage.retry"))
                log.warning("stage.retry", error=str(exc))
                if attempt < max_attempts:
                    time.sleep(self.cfg.retry.backoff_seconds * (2 ** (attempt - 1)))
                continue
            except Exception as exc:  # unexpected -> fail fast with record
                last_exc = exc
                traces.append(trace.finish(event="stage.error"))
                log.error("stage.failed", error=str(exc), exc_info=True)
                attempt_count = attempt
                break
            traces.append(trace.finish())
            record = StageRecord(
                stage=self.name,
                status="succeeded",
                attempts=attempt,
                started_at=started,
                finished_at=utc_now_iso(),
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            log.info("stage.done", duration_ms=record.duration_ms, attempts=attempt)
            return {**delta, "stages": {str(self.name): record}, "traces": traces}
        else:
            attempt_count = max_attempts
        assert last_exc is not None
        record = StageRecord(
            stage=self.name,
            status="failed",
            attempts=attempt_count,
            error=str(last_exc),
            started_at=started,
            finished_at=utc_now_iso(),
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
        self.log.error("stage.failed", attempts=attempt_count, error=str(last_exc))
        raise StageFailedError(str(self.name), attempt_count, last_exc) from last_exc

    def as_node(self) -> Any:
        def _node(state: PipelineState) -> dict[str, Any]:
            return self.run(state)

        _node.__name__ = str(self.name)
        return _node
