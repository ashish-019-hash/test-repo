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


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite tests/fixtures/expected/final.<fmt>.json from the current pipeline output",
    )


@pytest.fixture(scope="session")
def update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-golden"))
