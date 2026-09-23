"""LLM provider protocol and typed task definitions.

Providers never decide anything: they return *proposals* as Pydantic models, and
deterministic agent code scores, verifies (evidence must be a literal substring of
the chunk), and routes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

TResponse = TypeVar("TResponse", bound=BaseModel)


@dataclass(frozen=True)
class LLMTask:
    """A named structured-output task: prompt template + response model."""

    name: str
    prompt_file: str
    response_model: type[BaseModel]
    max_output_tokens: int = 2000

    def prompt_template(self) -> str:
        return (PROMPTS_DIR / self.prompt_file).read_text(encoding="utf-8")


@dataclass
class LLMCallResult:
    response: BaseModel
    cache_hit: bool = False
    system_fingerprint: str | None = None
    raw: dict[str, Any] | None = None


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def call(self, task: LLMTask, payload: dict[str, Any]) -> LLMCallResult:
        """Run `task` with `payload` (JSON-serialisable). Returns a `task.response_model` instance.

        Raises LLMTransientError for retryable failures and LLMPermanentError otherwise.
        """
        ...
