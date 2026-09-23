"""Tests for `doc_extractor.llm.cache`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from doc_extractor.llm.cache import CachingProvider, cache_key
from doc_extractor.llm.tasks import ATTRIBUTE_CANDIDATES_TASK, AttributeCandidatesResponse


class _CountingProvider:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def call(self, task: Any, payload: dict[str, Any]) -> Any:
        from doc_extractor.llm.base import LLMCallResult

        self.calls += 1
        return LLMCallResult(
            response=AttributeCandidatesResponse(candidates=[]), cache_hit=False, system_fingerprint="fp1"
        )


def test_cache_key_is_stable() -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    k1 = cache_key(ATTRIBUTE_CANDIDATES_TASK, payload, "gpt-4o", "2024-10-21")
    k2 = cache_key(ATTRIBUTE_CANDIDATES_TASK, dict(payload), "gpt-4o", "2024-10-21")
    assert k1 == k2


def test_cache_key_changes_with_payload() -> None:
    k1 = cache_key(ATTRIBUTE_CANDIDATES_TASK, {"a": 1}, "gpt-4o", "2024-10-21")
    k2 = cache_key(ATTRIBUTE_CANDIDATES_TASK, {"a": 2}, "gpt-4o", "2024-10-21")
    assert k1 != k2


def test_cache_key_changes_with_model() -> None:
    k1 = cache_key(ATTRIBUTE_CANDIDATES_TASK, {"a": 1}, "gpt-4o", "2024-10-21")
    k2 = cache_key(ATTRIBUTE_CANDIDATES_TASK, {"a": 1}, "gpt-4o-mini", "2024-10-21")
    assert k1 != k2


def test_cache_key_changes_with_api_version() -> None:
    k1 = cache_key(ATTRIBUTE_CANDIDATES_TASK, {"a": 1}, "gpt-4o", "2024-10-21")
    k2 = cache_key(ATTRIBUTE_CANDIDATES_TASK, {"a": 1}, "gpt-4o", "2024-08-01")
    assert k1 != k2


def test_miss_then_hit(tmp_path: Path) -> None:
    inner = _CountingProvider()
    provider = CachingProvider(inner, tmp_path, model="gpt-4o", api_version="2024-10-21")
    payload = {"source_text": "hello"}

    result1 = provider.call(ATTRIBUTE_CANDIDATES_TASK, payload)
    assert result1.cache_hit is False
    assert inner.calls == 1

    result2 = provider.call(ATTRIBUTE_CANDIDATES_TASK, payload)
    assert result2.cache_hit is True
    assert inner.calls == 1  # inner not called again
    assert result2.response.candidates == []
    assert result2.system_fingerprint == "fp1"


def test_cache_file_written_atomically(tmp_path: Path) -> None:
    inner = _CountingProvider()
    provider = CachingProvider(inner, tmp_path, model="gpt-4o", api_version="2024-10-21")
    provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "x"})

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    # No leftover temp files.
    assert not list(tmp_path.glob(".tmp-*"))


def test_name_matches_inner(tmp_path: Path) -> None:
    inner = _CountingProvider()
    provider = CachingProvider(inner, tmp_path, model="gpt-4o", api_version="2024-10-21")
    assert provider.name == "counting"
