"""Tests for `doc_extractor.llm.azure.AzureOpenAIProvider` using a fake OpenAI client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
import pytest

from doc_extractor.config import load_config
from doc_extractor.exceptions import LLMPermanentError, LLMTransientError
from doc_extractor.llm.azure import AzureOpenAIProvider
from doc_extractor.llm.settings import AzureSettings
from doc_extractor.llm.tasks import ATTRIBUTE_CANDIDATES_TASK


def _settings() -> AzureSettings:
    return AzureSettings.from_env(
        {
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "secret",
            "AZURE_OPENAI_API_VERSION": "2024-10-21",
            "AZURE_OPENAI_MODEL": "gpt-4o",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o-deploy",
        }
    )


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Completion:
    choices: list[_Choice]
    system_fingerprint: str | None = None


def _completion(content: str, system_fingerprint: str | None = "fp_test") -> _Completion:
    return _Completion(
        choices=[_Choice(message=_Message(content=content))], system_fingerprint=system_fingerprint
    )


def _http_request_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://example.openai.azure.com/chat/completions")
    return httpx.Response(status_code, request=request, json={"error": {"message": "boom"}})


@dataclass
class _FakeCompletions:
    calls: list[dict[str, Any]] = field(default_factory=list)
    script: list[Any] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@dataclass
class _FakeChat:
    completions: _FakeCompletions


@dataclass
class _FakeClient:
    chat: _FakeChat

    @classmethod
    def with_script(cls, script: list[Any]) -> _FakeClient:
        return cls(chat=_FakeChat(completions=_FakeCompletions(script=script)))


@pytest.fixture
def cfg() -> Any:
    return load_config(env={"LLM_PROVIDER": "azure"})


def test_valid_json_is_parsed(cfg: Any) -> None:
    client = _FakeClient.with_script([_completion('{"candidates": []}')])
    provider = AzureOpenAIProvider(cfg, _settings(), client=client)

    result = provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})

    assert result.response.candidates == []
    assert result.cache_hit is False
    assert result.system_fingerprint == "fp_test"
    assert len(client.chat.completions.calls) == 1
    assert client.chat.completions.calls[0]["model"] == "gpt-4o-deploy"


def test_invalid_json_reprompts_once_then_raises(cfg: Any) -> None:
    client = _FakeClient.with_script([_completion("not json at all"), _completion("still not json")])
    provider = AzureOpenAIProvider(cfg, _settings(), client=client)

    with pytest.raises(LLMPermanentError):
        provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})

    assert len(client.chat.completions.calls) == 2
    # The second call's messages must include the "previous answer was invalid" nudge.
    second_messages = client.chat.completions.calls[1]["messages"]
    assert "invalid" in second_messages[-1]["content"].lower()


def test_rate_limit_error_maps_to_transient(cfg: Any) -> None:
    response = _http_request_response(429)
    error = openai.RateLimitError("rate limited", response=response, body={})
    client = _FakeClient.with_script([error])
    provider = AzureOpenAIProvider(cfg, _settings(), client=client)

    with pytest.raises(LLMTransientError):
        provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})


def test_internal_server_error_maps_to_transient(cfg: Any) -> None:
    response = _http_request_response(500)
    error = openai.InternalServerError("boom", response=response, body={})
    client = _FakeClient.with_script([error])
    provider = AzureOpenAIProvider(cfg, _settings(), client=client)

    with pytest.raises(LLMTransientError):
        provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})


def test_permission_denied_maps_to_permanent(cfg: Any) -> None:
    response = _http_request_response(403)
    error = openai.PermissionDeniedError("nope", response=response, body={})
    client = _FakeClient.with_script([error])
    provider = AzureOpenAIProvider(cfg, _settings(), client=client)

    with pytest.raises(LLMPermanentError):
        provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})


def test_bad_request_mentioning_response_format_falls_back_to_json_object(cfg: Any) -> None:
    response = _http_request_response(400)
    error = openai.BadRequestError(
        "Invalid parameter: 'response_format' of type 'json_schema' is not supported.",
        response=response,
        body={},
    )
    client = _FakeClient.with_script([error, _completion('{"candidates": []}')])
    provider = AzureOpenAIProvider(cfg, _settings(), client=client)

    result = provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})

    assert result.response.candidates == []
    assert len(client.chat.completions.calls) == 2
    assert client.chat.completions.calls[0]["response_format"]["type"] == "json_schema"
    assert client.chat.completions.calls[1]["response_format"] == {"type": "json_object"}
    # In json_object mode the API no longer enforces the schema, so the prompt must carry it.
    system_prompt = client.chat.completions.calls[1]["messages"][0]["content"]
    assert '"source_text_quote"' in system_prompt
    assert '"required": ["raw_name", "source_text_quote", "sentence"]' in system_prompt

    # The fallback is remembered for subsequent calls on the same provider instance.
    client.chat.completions.script.append(_completion('{"candidates": []}'))
    provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "world"})
    assert client.chat.completions.calls[2]["response_format"] == {"type": "json_object"}


def test_json_schema_request_is_strict_with_every_property_required(cfg: Any) -> None:
    client = _FakeClient.with_script([_completion('{"candidates": []}')])
    provider = AzureOpenAIProvider(cfg, _settings(), client=client)

    provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})

    json_schema = client.chat.completions.calls[0]["response_format"]["json_schema"]
    assert json_schema["strict"] is True
    schema = json_schema["schema"]
    # Pydantic omits `candidates` (it has a default) from `required`; Azure strict mode
    # rejects that with "'required' ... must include every key in properties".
    assert schema["required"] == ["candidates"]
    assert schema["additionalProperties"] is False
    proposal = schema["$defs"]["CandidateProposal"]
    assert sorted(proposal["required"]) == ["raw_name", "sentence", "source_text_quote"]
    assert proposal["additionalProperties"] is False


def test_bad_request_unrelated_to_schema_raises_permanent(cfg: Any) -> None:
    response = _http_request_response(400)
    error = openai.BadRequestError(
        "Invalid parameter: 'temperature' must be between 0 and 2.", response=response, body={}
    )
    client = _FakeClient.with_script([error])
    provider = AzureOpenAIProvider(cfg, _settings(), client=client)

    with pytest.raises(LLMPermanentError):
        provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})


def test_model_label_records_settings_model(cfg: Any) -> None:
    client = _FakeClient.with_script([])
    provider = AzureOpenAIProvider(cfg, _settings(), client=client)
    assert provider.model_label == "gpt-4o"
