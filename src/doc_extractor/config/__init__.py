"""Configuration loading: default.yaml -> user YAML -> environment overrides -> validation."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import ConfigError
from doc_extractor.storage import ids
from doc_extractor.storage.canonical_json import dumps

PACKAGE_CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_CONFIG_DIR / "default.yaml"
ENV_PREFIX = "DOC_EXTRACTOR__"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """`DOC_EXTRACTOR__scoring__routing__accept=0.9` -> {"scoring": {"routing": {"accept": 0.9}}}."""
    result: dict[str, Any] = {}
    for key, raw in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].split("__")
        if not all(path):
            raise ConfigError(f"Malformed override variable: {key}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:  # pragma: no cover
            raise ConfigError(f"Cannot parse value of {key}: {exc}") from exc
        node = result
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    if env.get("LLM_PROVIDER"):
        result.setdefault("llm", {})["provider"] = env["LLM_PROVIDER"].strip().lower()
    return result


def _resolve_lexicon(rel: str, config_dir: Path | None) -> Path:
    candidates = [PACKAGE_CONFIG_DIR / rel]
    if config_dir is not None:
        candidates.insert(0, config_dir / rel)
    p = Path(rel)
    if p.is_absolute():
        candidates = [p]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise ConfigError(f"Lexicon file not found: {rel} (looked in {[str(c) for c in candidates]})")


def load_config(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> AppConfig:
    env = os.environ if env is None else env
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    config_dir: Path | None = None
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
        with p.open(encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ConfigError(f"Config file must be a mapping: {p}")
        data = _deep_merge(data, user)
        config_dir = p.resolve().parent
    data = _deep_merge(data, _env_overrides(env))
    try:
        cfg = AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
    except ConfigError:
        raise
    lexicon_paths = {
        name: _resolve_lexicon(rel, config_dir) for name, rel in cfg.lexicons.file_fields().items()
    }
    canonical = dumps(cfg.model_dump(mode="json"))
    contents = [lexicon_paths[k].read_text(encoding="utf-8") for k in sorted(lexicon_paths)]
    return cfg.model_copy(
        update={"lexicon_paths": lexicon_paths, "config_hash": ids.config_hash(canonical, contents)}
    )


__all__ = ["AppConfig", "DEFAULT_CONFIG_PATH", "ENV_PREFIX", "load_config"]
