"""Tests for `doc_extractor.llm.settings`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from doc_extractor.llm.settings import AzureSettings, load_dotenv_file


def test_from_env_empty_strings_are_missing() -> None:
    settings = AzureSettings.from_env(
        {
            "AZURE_OPENAI_ENDPOINT": "",
            "AZURE_OPENAI_API_KEY": "",
            "AZURE_OPENAI_API_VERSION": "",
            "AZURE_OPENAI_MODEL": "",
            "AZURE_OPENAI_DEPLOYMENT": "",
        }
    )
    assert settings.endpoint is None
    assert settings.api_key is None
    assert settings.api_version == "2024-10-21"
    assert settings.model is None
    assert settings.deployment is None
    assert not settings.is_complete
    assert settings.missing() == [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_MODEL",
        "AZURE_OPENAI_DEPLOYMENT",
    ]


def test_from_env_missing_entirely() -> None:
    settings = AzureSettings.from_env({})
    assert settings.missing() == [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_MODEL",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    assert not settings.is_complete


def test_deployment_falls_back_to_model_when_empty() -> None:
    settings = AzureSettings.from_env(
        {
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "secret",
            "AZURE_OPENAI_MODEL": "gpt-4o",
            "AZURE_OPENAI_DEPLOYMENT": "",
        }
    )
    assert settings.deployment == "gpt-4o"
    assert settings.is_complete
    assert settings.missing() == []


def test_deployment_direct_construction_falls_back_to_model() -> None:
    settings = AzureSettings(model="gpt-4o", deployment="")
    assert settings.deployment == "gpt-4o"


def test_is_complete_requires_endpoint_and_key() -> None:
    settings = AzureSettings.from_env(
        {"AZURE_OPENAI_MODEL": "gpt-4o", "AZURE_OPENAI_DEPLOYMENT": "gpt-4o-deploy"}
    )
    assert not settings.is_complete
    assert "AZURE_OPENAI_ENDPOINT" in settings.missing()
    assert "AZURE_OPENAI_API_KEY" in settings.missing()


def test_load_dotenv_file_populates_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AZURE_OPENAI_ENDPOINT=https://from-dotenv.openai.azure.com\n"
        "AZURE_OPENAI_API_KEY=from-dotenv-key\n"
        "AZURE_OPENAI_API_VERSION=2024-10-21\n"
        "AZURE_OPENAI_MODEL=gpt-4o\n"
        "AZURE_OPENAI_DEPLOYMENT=gpt-4o-deploy\n",
        encoding="utf-8",
    )
    for var in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_MODEL",
        "AZURE_OPENAI_DEPLOYMENT",
    ):
        monkeypatch.delenv(var, raising=False)

    loaded_path = load_dotenv_file(str(env_file))

    try:
        assert loaded_path == str(env_file)
        assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://from-dotenv.openai.azure.com"
        assert os.environ["AZURE_OPENAI_API_KEY"] == "from-dotenv-key"
        assert os.environ["AZURE_OPENAI_DEPLOYMENT"] == "gpt-4o-deploy"

        settings = AzureSettings.from_env(os.environ)
        assert settings.is_complete
    finally:
        for var in (
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_MODEL",
            "AZURE_OPENAI_DEPLOYMENT",
        ):
            monkeypatch.delenv(var, raising=False)


def test_load_dotenv_file_returns_none_when_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = load_dotenv_file()
    assert result is None


def test_load_dotenv_file_explicit_missing_path_is_still_the_path_used(tmp_path: Path) -> None:
    # Mirrors dotenv.load_dotenv's own behaviour: an explicit path is used as-is
    # (silently a no-op if the file does not exist), it is never replaced by None.
    missing = tmp_path / "does-not-exist.env"
    result = load_dotenv_file(str(missing))
    assert result == str(missing)


def test_load_dotenv_file_does_not_override_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("AZURE_OPENAI_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "from-shell")

    load_dotenv_file(str(env_file))

    assert os.environ["AZURE_OPENAI_MODEL"] == "from-shell"
