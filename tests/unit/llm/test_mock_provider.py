"""Tests for `doc_extractor.llm.mock` (ScriptedProvider, ReplayProvider)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import LLMPermanentError
from doc_extractor.llm.cache import cache_key
from doc_extractor.llm.mock import ReplayProvider, ScriptedProvider
from doc_extractor.llm.rules import RuleBasedProvider
from doc_extractor.llm.tasks import ATTRIBUTE_CANDIDATES_TASK, AttributeCandidatesResponse, CandidateProposal


def test_scripted_provider_pops_in_order() -> None:
    r1 = AttributeCandidatesResponse(candidates=[])
    r2 = AttributeCandidatesResponse(
        candidates=[CandidateProposal(raw_name="MAC Address", source_text_quote="MAC address", sentence=None)]
    )
    provider = ScriptedProvider({"attribute_candidates": [r1, r2]})

    out1 = provider.call(ATTRIBUTE_CANDIDATES_TASK, {"p": 1})
    out2 = provider.call(ATTRIBUTE_CANDIDATES_TASK, {"p": 2})

    assert out1.response is r1
    assert out2.response is r2
    assert provider.call_log == [("attribute_candidates", {"p": 1}), ("attribute_candidates", {"p": 2})]


def test_scripted_provider_raises_scripted_exception() -> None:
    provider = ScriptedProvider({"attribute_candidates": [LLMPermanentError("boom")]})
    with pytest.raises(LLMPermanentError, match="boom"):
        provider.call(ATTRIBUTE_CANDIDATES_TASK, {})


def test_scripted_provider_raises_when_exhausted() -> None:
    provider = ScriptedProvider({"attribute_candidates": []})
    with pytest.raises(LLMPermanentError):
        provider.call(ATTRIBUTE_CANDIDATES_TASK, {})


def test_scripted_provider_raises_when_task_never_scripted() -> None:
    provider = ScriptedProvider({})
    with pytest.raises(LLMPermanentError):
        provider.call(ATTRIBUTE_CANDIDATES_TASK, {})


def test_replay_provider_hits_fixture(tmp_path: Path, cfg: AppConfig) -> None:
    payload = {"source_text": "hello"}
    key = cache_key(ATTRIBUTE_CANDIDATES_TASK, payload, model=None, api_version="2024-10-21")
    fixture = tmp_path / f"{key}.json"
    response = AttributeCandidatesResponse(
        candidates=[CandidateProposal(raw_name="MAC Address", source_text_quote="MAC address", sentence=None)]
    )
    fixture.write_text(
        json.dumps({"task": "attribute_candidates", "response": response.model_dump(mode="json")}),
        encoding="utf-8",
    )

    provider = ReplayProvider(tmp_path, cfg, api_version="2024-10-21", model=None)
    result = provider.call(ATTRIBUTE_CANDIDATES_TASK, payload)

    assert result.cache_hit is True
    assert result.response.candidates[0].raw_name == "MAC Address"


def test_replay_provider_falls_back_to_rules_when_missing(tmp_path: Path, cfg: AppConfig) -> None:
    provider = ReplayProvider(tmp_path, cfg, strict=False)
    result = provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})
    assert result.response.candidates == []  # RuleBasedProvider with no scoring module -> empty
    assert result.cache_hit is False


def test_replay_provider_strict_raises_when_missing(tmp_path: Path, cfg: AppConfig) -> None:
    provider = ReplayProvider(tmp_path, cfg, strict=True)
    with pytest.raises(LLMPermanentError):
        provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})


def test_replay_provider_uses_custom_fallback(tmp_path: Path, cfg: AppConfig) -> None:
    fallback = RuleBasedProvider(cfg)
    provider = ReplayProvider(tmp_path, cfg, fallback=fallback)
    assert provider.fallback is fallback
    result = provider.call(ATTRIBUTE_CANDIDATES_TASK, {"source_text": "hello"})
    assert result.response.candidates == []


def test_replay_provider_default_name_is_mock(tmp_path: Path, cfg: AppConfig) -> None:
    provider = ReplayProvider(tmp_path, cfg)
    assert provider.name == "mock"
