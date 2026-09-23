from __future__ import annotations

import pytest

from doc_extractor.config import load_config
from doc_extractor.config.models import ALL_SIGNALS
from doc_extractor.exceptions import ConfigError


def test_default_config_loads_and_hashes():
    cfg = load_config(env={})
    assert cfg.scoring.routing.accept == 0.85
    assert cfg.scoring.routing.review == 0.50
    assert set(cfg.scoring.weights) == set(ALL_SIGNALS)
    assert len(cfg.config_hash) == 16
    assert len(cfg.lexicon_paths) == 7


def test_env_override_and_shortcut():
    cfg = load_config(env={"DOC_EXTRACTOR__scoring__routing__accept": "0.9", "LLM_PROVIDER": "rules"})
    assert cfg.scoring.routing.accept == 0.9
    assert cfg.llm.provider == "rules"


def test_config_hash_changes_with_override():
    a = load_config(env={})
    b = load_config(env={"DOC_EXTRACTOR__mapping__min_confidence": "0.7"})
    assert a.config_hash != b.config_hash


def test_user_yaml_merge(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("chunking:\n  strategy: by_page\n", encoding="utf-8")
    cfg = load_config(p, env={})
    assert cfg.chunking.strategy == "by_page"
    assert cfg.scoring.routing.accept == 0.85  # untouched default


@pytest.mark.parametrize(
    "env",
    [
        {"DOC_EXTRACTOR__scoring__routing__review": "0.9"},  # review >= accept
        {"DOC_EXTRACTOR__scoring__weights__negative.furniture": "0.5"},  # sign flip
        {"DOC_EXTRACTOR__reviewer__quality__weights__specificity": "0.5"},  # weights no longer sum to 1
        {"DOC_EXTRACTOR__reviewer__duplicate__possible_threshold": "0.95"},
    ],
)
def test_invalid_configs_raise(env):
    with pytest.raises(ConfigError):
        load_config(env=env)
