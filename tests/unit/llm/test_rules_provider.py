"""Tests for `doc_extractor.llm.rules.RuleBasedProvider`."""

from __future__ import annotations

from typing import Any

import pytest

from doc_extractor.config.models import AppConfig
from doc_extractor.llm.rules import RuleBasedProvider
from doc_extractor.llm.tasks import (
    ATTRIBUTE_CANDIDATES_TASK,
    BINDING_JUDGEMENT_TASK,
    ENTITY_PROPOSALS_TASK,
    ENTITY_QUALITY_TASK,
)


def test_attribute_candidates_empty_when_scoring_module_absent(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    # Force ImportError regardless of whether the scoring module has landed yet.
    monkeypatch.setitem(sys.modules, "doc_extractor.scoring.candidates", None)

    provider = RuleBasedProvider(cfg)
    result = provider.call(ATTRIBUTE_CANDIDATES_TASK, {"chunk": "anything"})
    assert result.response.candidates == []


def test_attribute_candidates_empty_when_payload_lacks_fields(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    fake_module = types.ModuleType("doc_extractor.scoring.candidates")

    def generate_candidates(chunk: Any, blocks: Any) -> list[Any]:
        raise AssertionError("should not be called: required args missing from payload")

    fake_module.generate_candidates = generate_candidates  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "doc_extractor.scoring.candidates", fake_module)

    provider = RuleBasedProvider(cfg)
    result = provider.call(ATTRIBUTE_CANDIDATES_TASK, {"chunk": "anything"})  # no "blocks" key
    assert result.response.candidates == []


def test_attribute_candidates_converts_generator_output(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    class _FakeCandidate:
        raw_name = "MAC Address"
        source_text = "a MAC address"
        sentence = "Each UNI shall have a MAC address."

    fake_module = types.ModuleType("doc_extractor.scoring.candidates")

    def generate_candidates(chunk: Any) -> list[Any]:
        assert chunk == "some-chunk"
        return [_FakeCandidate()]

    fake_module.generate_candidates = generate_candidates  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "doc_extractor.scoring.candidates", fake_module)

    provider = RuleBasedProvider(cfg)
    result = provider.call(ATTRIBUTE_CANDIDATES_TASK, {"chunk": "some-chunk"})
    assert len(result.response.candidates) == 1
    proposal = result.response.candidates[0]
    assert proposal.raw_name == "MAC Address"
    assert proposal.source_text_quote == "a MAC address"
    assert proposal.sentence == "Each UNI shall have a MAC address."


def test_entity_proposals_finds_whole_word_mentions(cfg: AppConfig) -> None:
    provider = RuleBasedProvider(cfg)
    text = "Each UNI SHALL have a physical medium. The EVC is bound to it."
    result = provider.call(ENTITY_PROPOSALS_TASK, {"source_text": text, "chunk_id": "chunk-doc-p1-0001"})
    names = {e.entity_name for e in result.response.entities}
    assert "UNI" in names
    assert "EVC" in names
    for e in result.response.entities:
        assert e.evidence_quote in text
        assert e.chunk_id == "chunk-doc-p1-0001"


def test_entity_proposals_matches_alias(cfg: AppConfig) -> None:
    provider = RuleBasedProvider(cfg)
    text = "The Ethernet Virtual Connection (EVC) spans two UNIs."
    result = provider.call(ENTITY_PROPOSALS_TASK, {"source_text": text, "chunk_id": "c1"})
    names = {e.entity_name for e in result.response.entities}
    assert "EVC" in names


def test_entity_proposals_no_false_positive_on_substring() -> None:
    from doc_extractor.config import load_config

    cfg = load_config(env={"LLM_PROVIDER": "rules"})
    provider = RuleBasedProvider(cfg)
    text = "The UNIQUE identifier is not related to any known term."
    result = provider.call(ENTITY_PROPOSALS_TASK, {"source_text": text, "chunk_id": "c1"})
    names = {e.entity_name for e in result.response.entities}
    assert "UNI" not in names


def test_entity_proposals_empty_source_text(cfg: AppConfig) -> None:
    provider = RuleBasedProvider(cfg)
    result = provider.call(ENTITY_PROPOSALS_TASK, {"source_text": "", "chunk_id": "c1"})
    assert result.response.entities == []


def test_entity_quality_returns_zero_deltas(cfg: AppConfig) -> None:
    provider = RuleBasedProvider(cfg)
    result = provider.call(
        ENTITY_QUALITY_TASK, {"entity_name": "UNI", "entity_type": "Interface", "evidence": []}
    )
    assert result.response.specificity_delta == 0.0
    assert result.response.real_world_delta == 0.0
    assert result.response.reason == "rules provider: no adjustment"


def test_binding_judgement_matches_pattern(cfg: AppConfig) -> None:
    provider = RuleBasedProvider(cfg)
    sentence = "Each UNI SHALL have a physical medium and a MAC address."
    result = provider.call(
        BINDING_JUDGEMENT_TASK,
        {
            "chunk_id": "c1",
            "source_text": sentence,
            "attributes": [{"attribute_id": "attr-1", "attribute_name": "MAC Address", "sentence": sentence}],
            "candidate_entities": ["UNI", "Subscriber"],
        },
    )
    [binding] = result.response.bindings
    assert binding.attribute_id == "attr-1"
    assert binding.entity_name == "UNI"
    assert binding.evidence_quote == sentence


def test_binding_judgement_no_match_returns_null(cfg: AppConfig) -> None:
    provider = RuleBasedProvider(cfg)
    sentence = "The document describes general concepts without any binding verb."
    result = provider.call(
        BINDING_JUDGEMENT_TASK,
        {
            "chunk_id": "c1",
            "source_text": sentence,
            "attributes": [
                {"attribute_id": "attr-1", "attribute_name": "Concept", "sentence": sentence},
                {"attribute_id": "attr-2", "attribute_name": "Other", "sentence": ""},
            ],
            "candidate_entities": ["UNI", "Subscriber"],
        },
    )
    assert [b.attribute_id for b in result.response.bindings] == ["attr-1", "attr-2"]
    assert all(b.entity_name is None and b.evidence_quote is None for b in result.response.bindings)


def test_provider_name() -> None:
    from doc_extractor.config import load_config

    cfg = load_config(env={"LLM_PROVIDER": "rules"})
    provider = RuleBasedProvider(cfg)
    assert provider.name == "rules"
