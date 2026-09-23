"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_extractor.config import load_config
from doc_extractor.config.models import AppConfig

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    """Default config with the rules provider forced (no network in tests)."""
    return load_config(env={"LLM_PROVIDER": "rules"})


@pytest.fixture(scope="session")
def telecom_spec_md(fixtures_dir: Path) -> Path:
    return fixtures_dir / "telecom_spec.md"
