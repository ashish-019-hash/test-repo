"""Azure OpenAI-backed `LLMProvider`.

Structured outputs only: every response is validated against `task.response_model`
before it reaches an agent. Azure API versions vary in `json_schema` support, so a
`BadRequestError` that mentions the schema mode falls back to plain `json_object`
mode (and that fallback is remembered for the life of the provider instance).
"""

from __future__ import annotations

import json
from typing import Any

import openai
from pydantic import BaseModel

from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import ConfigError, LLMPermanentError, LLMTransientError
from doc_extractor.llm.base import LLMCallResult, LLMTask
from doc_extractor.llm.settings import AzureSettings
from doc_extractor.observability.logging import get_logger

_SYSTEM_PROMPT = (
    "You are a deterministic JSON-generating assistant. Respond with a single JSON "
    "object matching the requested schema exactly. Never include markdown fences, "
    "commentary, or text outside the JSON object."
)

_TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _mentions_schema_rejection(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "json_schema" in message


class AzureOpenAIProvider:
    """`LLMProvider` backed by `openai.AzureOpenAI`."""

    name = "azure"

    def __init__(
        self,
        cfg: AppConfig,
        settings: AzureSettings,
        client: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.settings = settings
        self.model_label = settings.model
        self._use_json_object = False
        if client is not None:
            self.client = client
        else:
            if not settings.endpoint or not settings.api_key or not settings.deployment:
                raise ConfigError(
                    "AzureOpenAIProvider requires a complete AzureSettings (endpoint, api_key, deployment)"
                )
            self.client = openai.AzureOpenAI(
                azure_endpoint=settings.endpoint,
                api_key=settings.api_key,
                api_version=settings.api_version,
                timeout=cfg.llm.timeout_seconds,
            )

    def call(self, task: LLMTask, payload: dict[str, Any]) -> LLMCallResult:
        system = {"role": "system", "content": _SYSTEM_PROMPT}
        user = {
            "role": "user",
            "content": task.prompt_template().format(
                payload_json=json.dumps(payload, sort_keys=True, ensure_ascii=False)
            ),
        }
        messages: list[dict[str, str]] = [system, user]

        get_logger().info("llm.call", provider=self.name, task=task.name, model=self.model_label)
        completion = self._create(task, messages)
        response, error = self._parse(task, completion)
        if error is None:
            return LLMCallResult(
                response=response,  # type: ignore[arg-type]
                cache_hit=False,
                system_fingerprint=getattr(completion, "system_fingerprint", None),
            )

        retry_messages = [
            *messages,
            {
                "role": "user",
                "content": f"Your previous answer was invalid: {error}. Return only valid JSON.",
            },
        ]
        completion = self._create(task, retry_messages)
        response, error = self._parse(task, completion)
        if error is None:
            return LLMCallResult(
                response=response,  # type: ignore[arg-type]
                cache_hit=False,
                system_fingerprint=getattr(completion, "system_fingerprint", None),
            )
        raise LLMPermanentError(f"[{task.name}] invalid JSON after retry: {error}")

    # -- internals -----------------------------------------------------------------

    def _response_format(self, task: LLMTask) -> dict[str, Any]:
        if self._use_json_object:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": task.name,
                "schema": task.response_model.model_json_schema(),
                "strict": True,
            },
        }

    def _invoke(
        self, messages: list[dict[str, str]], response_format: dict[str, Any], *, max_tokens: int
    ) -> Any:
        try:
            return self.client.chat.completions.create(
                model=self.settings.deployment,
                temperature=self.cfg.llm.temperature,
                seed=self.cfg.llm.seed,
                max_tokens=max_tokens,
                messages=messages,
                response_format=response_format,
            )
        except openai.BadRequestError:
            raise
        except _TRANSIENT_EXCEPTIONS as exc:
            raise LLMTransientError(str(exc)) from exc
        except openai.APIStatusError as exc:
            raise LLMPermanentError(str(exc)) from exc

    def _create(self, task: LLMTask, messages: list[dict[str, str]]) -> Any:
        response_format = self._response_format(task)
        try:
            return self._invoke(messages, response_format, max_tokens=task.max_output_tokens)
        except openai.BadRequestError as exc:
            if self._use_json_object or not _mentions_schema_rejection(exc):
                raise LLMPermanentError(str(exc)) from exc
            self._use_json_object = True
            fallback_format = self._response_format(task)
            try:
                return self._invoke(messages, fallback_format, max_tokens=task.max_output_tokens)
            except openai.BadRequestError as exc2:
                raise LLMPermanentError(str(exc2)) from exc2

    def _parse(self, task: LLMTask, completion: Any) -> tuple[BaseModel | None, str | None]:
        content = completion.choices[0].message.content
        try:
            return task.response_model.model_validate_json(content or ""), None
        except ValueError as exc:
            return None, str(exc)


__all__ = ["AzureOpenAIProvider"]
