"""Test-only `LLMProvider` implementations: fully scripted, or replayed from fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import LLMPermanentError
from doc_extractor.llm.base import LLMCallResult, LLMProvider, LLMTask
from doc_extractor.llm.cache import cache_key
from doc_extractor.llm.rules import RuleBasedProvider
from doc_extractor.llm.settings import AzureSettings


class ScriptedProvider:
    """Serves pre-scripted responses (or raises pre-scripted exceptions), in order."""

    def __init__(self, responses: dict[str, list[BaseModel | Exception]], name: str = "mock") -> None:
        self.name = name
        self._responses: dict[str, list[BaseModel | Exception]] = {k: list(v) for k, v in responses.items()}
        self.call_log: list[tuple[str, dict[str, Any]]] = []

    def call(self, task: LLMTask, payload: dict[str, Any]) -> LLMCallResult:
        self.call_log.append((task.name, payload))
        queue = self._responses.get(task.name)
        if not queue:
            raise LLMPermanentError(f"ScriptedProvider exhausted for task {task.name!r}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMCallResult(response=item, cache_hit=False, system_fingerprint=None)


class ReplayProvider:
    """Serves cached/recorded `LLMCallResult`s by the same key `cache.py` uses.

    Missing fixtures either raise (`strict=True`) or fall through to `fallback`
    (a `RuleBasedProvider` by default), so replay-mode tests never touch the network.
    """

    def __init__(
        self,
        fixtures_dir: Path | str,
        cfg: AppConfig,
        strict: bool = False,
        fallback: LLMProvider | None = None,
        name: str = "mock",
        model: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self.name = name
        self.fixtures_dir = Path(fixtures_dir)
        self.cfg = cfg
        self.strict = strict
        self.fallback: LLMProvider = fallback or RuleBasedProvider(cfg)
        settings = AzureSettings.from_env({})
        self.model = model if model is not None else settings.model
        self.api_version = api_version if api_version is not None else settings.api_version

    def call(self, task: LLMTask, payload: dict[str, Any]) -> LLMCallResult:
        key = cache_key(task, payload, self.model, self.api_version)
        path = self.fixtures_dir / f"{key}.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            response = task.response_model.model_validate(data["response"])
            return LLMCallResult(
                response=response,
                cache_hit=True,
                system_fingerprint=data.get("system_fingerprint"),
            )
        if self.strict:
            raise LLMPermanentError(f"No replay fixture for task {task.name!r} at {path}")
        return self.fallback.call(task, payload)


__all__ = ["ScriptedProvider", "ReplayProvider"]
