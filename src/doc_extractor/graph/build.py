"""LangGraph wiring: a linear `StateGraph` over the nine agents.

START -> document_processing -> chunking -> attribute_extraction -> attribute_storage
      -> entity_generation -> entity_normalization -> entity_reviewer -> attribute_mapping
      -> final_output -> END

Retries live inside `BaseAgent.run`, so no node-level `RetryPolicy` is attached and
every agent stays testable without the graph. Checkpoints go to SQLite in the output
directory; the thread id is derived from the document and config hashes so re-running
the same input with the same configuration resumes the same thread.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from doc_extractor.agents import AGENT_REGISTRY
from doc_extractor.agents.base import BaseAgent
from doc_extractor.config.models import AppConfig
from doc_extractor.llm.base import LLMProvider
from doc_extractor.schemas import attribute, chunk, common, document, entity, mapping, output, review, state
from doc_extractor.schemas.state import STAGE_ORDER, PipelineState, StageName

CHECKPOINT_FILE = "checkpoints.sqlite"

# Pydantic models that may appear in checkpointed state. Registering them keeps the
# checkpoint serializer strict (no arbitrary-type deserialization) and silences the
# "unregistered type" warnings LangGraph emits otherwise.
_SCHEMA_MODULES = (attribute, chunk, common, document, entity, mapping, output, review, state)


def _exported_types(module: Any) -> list[type]:
    found: list[type] = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and obj.__module__ == module.__name__:
            if issubclass(obj, BaseModel) or issubclass(obj, Enum):
                found.append(obj)
    return found


ALLOWED_CHECKPOINT_TYPES: tuple[type, ...] = tuple(t for m in _SCHEMA_MODULES for t in _exported_types(m))


def build_agents(
    cfg: AppConfig,
    provider: LLMProvider,
    log: Any | None = None,
    overrides: Mapping[StageName, type[BaseAgent]] | None = None,
) -> dict[StageName, BaseAgent]:
    """Instantiate one agent per stage, in `STAGE_ORDER`.

    `overrides` swaps an agent class for a given stage (used by tests to inject failures).
    """
    result: dict[StageName, BaseAgent] = {}
    for stage in STAGE_ORDER:
        cls = (overrides or {}).get(stage) or AGENT_REGISTRY[stage]
        result[stage] = cls(cfg, provider, log=log)
    return result


def build_state_graph(agents: Mapping[StageName, BaseAgent]) -> StateGraph:
    """Return the uncompiled linear graph."""
    graph: StateGraph = StateGraph(PipelineState)
    for stage in STAGE_ORDER:
        graph.add_node(str(stage), agents[stage].as_node())
    graph.add_edge(START, str(STAGE_ORDER[0]))
    for prev, nxt in zip(STAGE_ORDER, STAGE_ORDER[1:], strict=False):
        graph.add_edge(str(prev), str(nxt))
    graph.add_edge(str(STAGE_ORDER[-1]), END)
    return graph


def compile_graph(agents: Mapping[StageName, BaseAgent], checkpointer: BaseCheckpointSaver | None) -> Any:
    return build_state_graph(agents).compile(checkpointer=checkpointer)


@contextmanager
def open_checkpointer(out_dir: str | Path) -> Iterator[SqliteSaver]:
    """SQLite checkpointer stored at `<out_dir>/checkpoints.sqlite`."""
    path = Path(out_dir) / CHECKPOINT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_CHECKPOINT_TYPES))
    finally:
        conn.close()


def thread_id(document_id: str, config_hash: str) -> str:
    return f"{document_id}-{config_hash}"


__all__ = [
    "ALLOWED_CHECKPOINT_TYPES",
    "CHECKPOINT_FILE",
    "build_agents",
    "build_state_graph",
    "compile_graph",
    "open_checkpointer",
    "thread_id",
]
