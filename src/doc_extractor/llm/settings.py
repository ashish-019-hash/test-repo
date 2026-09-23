"""Azure OpenAI settings loaded strictly from environment variables.

The user requires exactly five env vars (see `.env.example`): endpoint, api key,
api version, model, deployment. Empty strings are treated as missing so that an
`.env` file with blank placeholders behaves the same as unset variables.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

ENV_VARS: tuple[str, ...] = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_MODEL",
    "AZURE_OPENAI_DEPLOYMENT",
)

DEFAULT_API_VERSION = "2024-10-21"


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


class AzureSettings(BaseModel):
    """Typed view over the five `AZURE_OPENAI_*` environment variables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str | None = None
    api_key: str | None = None
    api_version: str = DEFAULT_API_VERSION
    model: str | None = None
    deployment: str | None = None

    @field_validator("endpoint", "api_key", "model", "deployment", mode="before")
    @classmethod
    def _empty_to_none(cls, v: str | None) -> str | None:
        return _clean(v)

    @field_validator("api_version", mode="before")
    @classmethod
    def _default_api_version(cls, v: str | None) -> str:
        return _clean(v) or DEFAULT_API_VERSION

    @model_validator(mode="after")
    def _deployment_falls_back_to_model(self) -> AzureSettings:
        if not self.deployment and self.model:
            object.__setattr__(self, "deployment", self.model)
        return self

    @property
    def is_complete(self) -> bool:
        return bool(self.endpoint) and bool(self.api_key) and bool(self.deployment or self.model)

    def missing(self) -> list[str]:
        """Env var names that are empty/unset, in the canonical `ENV_VARS` order."""
        out: list[str] = []
        if not self.endpoint:
            out.append("AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            out.append("AZURE_OPENAI_API_KEY")
        if not self.api_version:
            out.append("AZURE_OPENAI_API_VERSION")
        if not self.model:
            out.append("AZURE_OPENAI_MODEL")
        if not (self.deployment or self.model):
            out.append("AZURE_OPENAI_DEPLOYMENT")
        return out

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AzureSettings:
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            endpoint=source.get("AZURE_OPENAI_ENDPOINT"),
            api_key=source.get("AZURE_OPENAI_API_KEY"),
            api_version=source.get("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION,
            model=source.get("AZURE_OPENAI_MODEL"),
            deployment=source.get("AZURE_OPENAI_DEPLOYMENT"),
        )


def load_dotenv_file(path: str | None = None) -> str | None:
    """Load a `.env` file (without overriding already-exported shell vars).

    Returns the path that was actually loaded, or `None` if no `.env` file was found.
    """
    target = path or find_dotenv(usecwd=True)
    if not target:
        return None
    load_dotenv(target, override=False)
    return target


__all__ = ["ENV_VARS", "DEFAULT_API_VERSION", "AzureSettings", "load_dotenv_file"]
