"""Tests for `doc_extractor.llm.factory.build_provider`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from doc_extractor.config import load_config
from doc_extractor.exceptions import ConfigError
from doc_extractor.llm.azure import AzureOpenAIProvider
from doc_extractor.llm.cache import CachingProvider
from doc_extractor.llm.factory import build_provider, provider_model_label
from doc_extractor.llm.rules import RuleBasedProvider

_ALL_AZURE_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
    "AZURE_OPENAI_API_KEY": "secret",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "AZURE_OPENAI_MODEL": "gpt-4o",
    "AZURE_OPENAI_DEPLOYMENT": "gpt-4o-deploy",
}


class _FakeAzureOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_auto_falls_back_to_rules_when_env_missing() -> None:
    from doc_extractor.observability import get_logger

    get_logger()  # ensure structlog is configured once before capture_logs() takes over
    cfg = load_config(env={"LLM_PROVIDER": "auto"})
    with capture_logs() as logs:
        provider = build_provider(cfg, env={})
    assert isinstance(provider, RuleBasedProvider)
    warnings = [e for e in logs if e.get("log_level") == "warning"]
    assert any(e.get("event") == "llm.fallback" for e in warnings)
    fallback_log = next(e for e in warnings if e.get("event") == "llm.fallback")
    assert fallback_log["provider"] == "rules"
    assert fallback_log["reason"] == "missing_env"
    assert "AZURE_OPENAI_ENDPOINT" in fallback_log["missing"]


def test_auto_uses_azure_when_env_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("doc_extractor.llm.azure.openai.AzureOpenAI", _FakeAzureOpenAI)
    cfg = load_config(env={"LLM_PROVIDER": "auto"})

    provider = build_provider(cfg, env=_ALL_AZURE_ENV, cache_dir=tmp_path)

    assert isinstance(provider, CachingProvider)
    assert isinstance(provider.inner, AzureOpenAIProvider)
    assert isinstance(provider.inner.client, _FakeAzureOpenAI)
    assert provider_model_label(provider) == "gpt-4o"


def test_auto_without_cache_dir_returns_bare_azure_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("doc_extractor.llm.azure.openai.AzureOpenAI", _FakeAzureOpenAI)
    cfg = load_config(env={"LLM_PROVIDER": "auto"})

    provider = build_provider(cfg, env=_ALL_AZURE_ENV, cache_dir=None)

    assert isinstance(provider, AzureOpenAIProvider)
    assert provider_model_label(provider) == "gpt-4o"


def test_explicit_azure_missing_env_raises_config_error() -> None:
    cfg = load_config(env={"LLM_PROVIDER": "azure"})
    with pytest.raises(ConfigError):
        build_provider(cfg, env={})


def test_explicit_rules_via_load_config() -> None:
    cfg = load_config(env={"LLM_PROVIDER": "rules"})
    provider = build_provider(cfg, env={})
    assert isinstance(provider, RuleBasedProvider)


def test_mock_provider_without_replay_dir_raises() -> None:
    cfg = load_config(env={"LLM_PROVIDER": "mock"})
    with pytest.raises(ConfigError):
        build_provider(cfg, env={}, cache_dir=None)


def test_mock_provider_with_cache_dir(tmp_path: Path) -> None:
    from doc_extractor.llm.mock import ReplayProvider

    cfg = load_config(env={"LLM_PROVIDER": "mock"})
    provider = build_provider(cfg, env={}, cache_dir=tmp_path)
    assert isinstance(provider, ReplayProvider)


def test_provider_model_label_none_for_rules() -> None:
    cfg = load_config(env={"LLM_PROVIDER": "rules"})
    provider = build_provider(cfg, env={})
    assert provider_model_label(provider) is None
