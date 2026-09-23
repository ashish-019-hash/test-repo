"""Per-agent trace records and timing helpers.

LangSmith: nothing custom is needed. LangGraph reads LANGSMITH_TRACING, LANGSMITH_API_KEY
and LANGSMITH_PROJECT from the environment; this module never touches them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from doc_extractor.schemas.state import StageName, TraceRecord


@dataclass
class TraceCollector:
    """Accumulates trace data for one agent run; `finish` emits a TraceRecord."""

    stage: StageName
    llm_calls: int = 0
    cache_hits: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter)

    def count(self, key: str, n: int = 1) -> None:
        self.data[key] = int(self.data.get(key, 0)) + n

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def record_llm(self, cache_hit: bool) -> None:
        self.llm_calls += 1
        if cache_hit:
            self.cache_hits += 1

    def finish(self, event: str = "stage.done") -> TraceRecord:
        return TraceRecord(
            stage=self.stage,
            event=event,
            data=dict(sorted(self.data.items())),
            llm_calls=self.llm_calls,
            cache_hits=self.cache_hits,
            elapsed_ms=int((time.perf_counter() - self._t0) * 1000),
        )


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
