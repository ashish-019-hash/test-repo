"""Deterministic on-disk cache for `LLMProvider.call`.

Cache files double as replay fixtures for `mock.ReplayProvider`, so the key must
be stable across runs: it depends only on the task, the exact prompt template
text, the payload's canonical JSON, and the model/api-version pair.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from doc_extractor.llm.base import LLMCallResult, LLMProvider, LLMTask
from doc_extractor.observability.logging import get_logger
from doc_extractor.storage import canonical_json


def cache_key(task: LLMTask, payload: dict[str, Any], model: str | None, api_version: str | None) -> str:
    prompt_hash = hashlib.sha256(task.prompt_template().encode("utf-8")).hexdigest()
    payload_json = canonical_json.dumps(payload)
    parts = "|".join([task.name, prompt_hash, payload_json, str(model or ""), str(api_version or "")])
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


class CachingProvider:
    """Wraps another `LLMProvider`, memoizing `call` results as JSON files on disk."""

    def __init__(
        self, inner: LLMProvider, cache_dir: Path, model: str | None, api_version: str | None
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.api_version = api_version

    def call(self, task: LLMTask, payload: dict[str, Any]) -> LLMCallResult:
        key = cache_key(task, payload, self.model, self.api_version)
        path = self.cache_dir / f"{key}.json"
        if path.is_file():
            get_logger().info("llm.cache_hit", provider=self.name, task=task.name, key=key)
            data = json.loads(path.read_text(encoding="utf-8"))
            response = task.response_model.model_validate(data["response"])
            return LLMCallResult(
                response=response,
                cache_hit=True,
                system_fingerprint=data.get("system_fingerprint"),
            )

        result = self.inner.call(task, payload)
        self._write(path, task, result)
        return LLMCallResult(
            response=result.response,
            cache_hit=False,
            system_fingerprint=result.system_fingerprint,
            raw=result.raw,
        )

    def _write(self, path: Path, task: LLMTask, result: LLMCallResult) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": task.name,
            "model": self.model,
            "response": result.response.model_dump(mode="json"),
            "system_fingerprint": result.system_fingerprint,
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        fd, tmp_name = tempfile.mkstemp(dir=self.cache_dir, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)


__all__ = ["cache_key", "CachingProvider"]
