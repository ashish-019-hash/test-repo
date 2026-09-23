"""Selects and wires up the configured `LLMProvider`.

`cfg.llm.provider` (or the `LLM_PROVIDER` env shortcut baked into `cfg` by
`load_config`) decides which provider is built: `rules`, `mock`, `azure`, or
`auto` (Azure when fully configured, otherwise a logged fallback to `rules`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import ConfigError
from doc_extractor.llm.azure import AzureOpenAIProvider
from doc_extractor.llm.base import LLMProvider
from doc_extractor.llm.cache import CachingProvider
from doc_extractor.llm.mock import ReplayProvider
from doc_extractor.llm.rules import RuleBasedProvider
from doc_extractor.llm.settings import AzureSettings
from doc_extractor.observability import get_logger


def _wrap_azure(cfg: AppConfig, settings: AzureSettings, cache_dir: Path | None) -> LLMProvider:
    provider = AzureOpenAIProvider(cfg, settings)
    resolved = cfg.llm.cache_dir or cache_dir
    if resolved:
        return CachingProvider(
            provider, Path(resolved), model=settings.model, api_version=settings.api_version
        )
    return provider


def build_provider(
    cfg: AppConfig,
    env: Mapping[str, str] | None = None,
    cache_dir: Path | None = None,
    log: Any | None = None,
) -> LLMProvider:
    env = os.environ if env is None else env
    logger = log or get_logger()
    provider_name = cfg.llm.provider

    if provider_name == "rules":
        return RuleBasedProvider(cfg)

    if provider_name == "mock":
        fixtures = cfg.llm.replay_dir or cache_dir
        if fixtures is None:
            raise ConfigError("llm.provider='mock' requires llm.replay_dir or a cache_dir")
        return ReplayProvider(Path(fixtures), cfg, strict=cfg.llm.replay_strict)

    if provider_name == "azure":
        settings = AzureSettings.from_env(env)
        if not settings.is_complete:
            missing = settings.missing()
            raise ConfigError(f"llm.provider='azure' requires env vars: {', '.join(missing)}")
        return _wrap_azure(cfg, settings, cache_dir)

    if provider_name == "auto":
        settings = AzureSettings.from_env(env)
        if settings.is_complete:
            return _wrap_azure(cfg, settings, cache_dir)
        missing = settings.missing()
        logger.warning("llm.fallback", provider="rules", reason="missing_env", missing=missing)
        return RuleBasedProvider(cfg)

    raise ConfigError(f"Unknown llm.provider: {provider_name!r}")


def provider_model_label(provider: LLMProvider) -> str | None:
    """The Azure model name to record in `run.json`, if the provider chain uses Azure."""
    inner: Any = getattr(provider, "inner", provider)
    return getattr(inner, "model_label", None)


__all__ = ["build_provider", "provider_model_label"]
