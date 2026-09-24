"""Agents consuming Azure structured responses, driven by a `ScriptedProvider(name="azure")`.

These tests pin the response contracts in `doc_extractor.llm.tasks` to the fields the
agents actually read, so a mismatch can never silently drop model output again. No
network: the provider is scripted.
"""

from __future__ import annotations

from typing import Any

import pytest

from doc_extractor.agents.attribute_extraction import AttributeExtractionAgent
from doc_extractor.agents.entity_generation import EntityGenerationAgent
from doc_extractor.agents.entity_normalization import EntityNormalizationAgent
from doc_extractor.agents.entity_reviewer import EntityReviewerAgent
from doc_extractor.exceptions import LLMTransientError
from doc_extractor.llm import tasks
from doc_extractor.llm.mock import ScriptedProvider
from doc_extractor.schemas.state import StageName, StageRecord
from helpers_c import ALL_CHUNKS, CHUNK_INTRO, CHUNK_SUBSCRIBER, CHUNK_UNI_PROSE, base_state


def _azure(responses: dict[str, list[Any]]) -> ScriptedProvider:
    return ScriptedProvider(responses, name="azure")


def _empty_entities(n: int) -> list[Any]:
    return [tasks.EntityProposalsResponse(entities=[]) for _ in range(n)]


# --------------------------------------------------------------------------------------
# Agent 5: entity proposals
# --------------------------------------------------------------------------------------
def test_entity_generation_consumes_grounded_azure_proposals(cfg) -> None:
    proposals = _empty_entities(len(ALL_CHUNKS))
    proposals[ALL_CHUNKS.index(CHUNK_INTRO)] = tasks.EntityProposalsResponse(
        entities=[
            # merges into the existing binding entity: new evidence text on a known chunk
            tasks.EntityProposal(
                entity_name="Subscriber",
                entity_type="Party",
                chunk_id=CHUNK_INTRO.chunk_id,
                evidence_quote="delivered by a Service Provider to a Subscriber",
            ),
        ]
    )
    proposals[ALL_CHUNKS.index(CHUNK_SUBSCRIBER)] = tasks.EntityProposalsResponse(
        entities=[
            # new, grounded, but a single LLM mention -> reaches the confidence gate and is
            # dropped there (0.7 * 1/3 < min_confidence), not by the evidence filter
            tasks.EntityProposal(
                entity_name="Operator",
                entity_type="Party",
                chunk_id=CHUNK_SUBSCRIBER.chunk_id,
                evidence_quote="terminated by the operator",
            ),
            # hallucinated quote -> rejected
            tasks.EntityProposal(
                entity_name="Billing Account",
                entity_type="Party",
                chunk_id=CHUNK_SUBSCRIBER.chunk_id,
                evidence_quote="billing account of the subscriber",
            ),
            # quote copied from another chunk -> rejected (chunk id mismatch)
            tasks.EntityProposal(
                entity_name="Port",
                entity_type="Resource",
                chunk_id=CHUNK_UNI_PROSE.chunk_id,
                evidence_quote="port speed",
            ),
        ]
    )
    provider = _azure({"entity_proposals": proposals})
    agent = EntityGenerationAgent(cfg, provider)
    delta = agent.run(base_state(StageName.attribute_storage))

    assert [name for name, _ in provider.call_log] == ["entity_proposals"] * len(ALL_CHUNKS)
    names = {e.entity_name for e in delta["entities"]}
    assert names == {"EVC", "Service", "Service Provider", "Subscriber", "UNI"}
    subscriber = next(e for e in delta["entities"] if e.entity_name == "Subscriber")
    assert subscriber.origin == "binding"  # binding still wins the merge
    assert "delivered by a Service Provider to a Subscriber" in [
        ev.text for ev in subscriber.supporting_evidence
    ]
    trace = delta["traces"][-1]
    assert trace.llm_calls == len(ALL_CHUNKS)
    assert trace.data.get("evidence_rejected") == 2
    assert trace.data.get("entity_below_min_confidence") == 1


def test_entity_generation_azure_transient_error_is_retried(cfg) -> None:
    fast = cfg.model_copy(update={"retry": cfg.retry.model_copy(update={"backoff_seconds": 0.0})})
    responses = [LLMTransientError("429"), *_empty_entities(len(ALL_CHUNKS))]
    provider = _azure({"entity_proposals": responses})
    delta = EntityGenerationAgent(fast, provider).run(base_state(StageName.attribute_storage))
    assert delta["stages"][str(StageName.entity_generation)].attempts == 2


# --------------------------------------------------------------------------------------
# Agent 7: quality adjustment
# --------------------------------------------------------------------------------------
def _state_through_normalization(cfg):
    provider = ScriptedProvider({}, name="rules")
    state = base_state(StageName.attribute_storage)
    state.update(EntityGenerationAgent(cfg, provider).run(state))
    state["stages"][str(StageName.entity_generation)] = StageRecord(
        stage=StageName.entity_generation, attempts=1, status="succeeded"
    )
    state.update(EntityNormalizationAgent(cfg, provider).run(state))
    state["stages"][str(StageName.entity_normalization)] = StageRecord(
        stage=StageName.entity_normalization, attempts=1, status="succeeded"
    )
    return state


def test_reviewer_applies_both_azure_deltas_and_reason(cfg) -> None:
    state = _state_through_normalization(cfg)
    n = len(state["entities"])
    baseline = EntityReviewerAgent(cfg, ScriptedProvider({}, name="rules")).run(state)

    quality = [
        tasks.EntityQualityResponse(specificity_delta=-0.1, real_world_delta=0.9, reason="model says so")
        for _ in range(n)
    ]
    provider = _azure({"entity_quality": quality})
    delta = EntityReviewerAgent(cfg, provider).run(state)
    assert len(provider.call_log) == n

    max_adj = cfg.reviewer.quality.llm_adjustment_max
    base_by_id = {d.entity_id: d for d in baseline["review_decisions"]}
    for decision in delta["review_decisions"]:
        before = {c.criterion: c for c in base_by_id[decision.entity_id].criterion_scores}
        after = {c.criterion: c for c in decision.criterion_scores}
        spec_b, spec_a = before["specificity"], after["specificity"]
        rw_b, rw_a = before["real_world_correspondence"], after["real_world_correspondence"]
        assert spec_a.score == pytest.approx(max(0.0, spec_b.score - 0.1))
        assert rw_a.score == pytest.approx(min(1.0, rw_b.score + max_adj))  # clamped to +0.2
        assert "llm_adjustment=-0.10 (model says so)" in spec_a.reason
        assert f"llm_adjustment=+{max_adj:.2f} (model says so)" in rw_a.reason
        # untouched criteria are unchanged
        for name in before:
            if name not in ("specificity", "real_world_correspondence"):
                assert after[name] == before[name]


# --------------------------------------------------------------------------------------
# Agent 3: attribute candidate proposals
# --------------------------------------------------------------------------------------
def test_attribute_extraction_grounds_azure_sentence_and_quote(cfg, telecom_spec_md) -> None:
    from doc_extractor.chunking import chunk_document
    from doc_extractor.ingest import load_document

    doc = load_document(telecom_spec_md, cfg)
    chunks = chunk_document(doc, cfg)
    target = next(c for c in chunks if "5 working days" in c.source_text)
    responses = []
    for chunk in chunks:
        if chunk is target:
            responses.append(
                tasks.AttributeCandidatesResponse(
                    candidates=[
                        # grounded quote, hallucinated sentence -> evidence falls back to the quote
                        tasks.CandidateProposal(
                            raw_name="Provisioning Lead Time",
                            source_text_quote="5 working days",
                            sentence="Provisioning always completes within five business days.",
                        ),
                        # hallucinated quote -> dropped entirely
                        tasks.CandidateProposal(
                            raw_name="Activation Fee",
                            source_text_quote="an activation fee of 50 EUR",
                            sentence=None,
                        ),
                    ]
                )
            )
        else:
            responses.append(tasks.AttributeCandidatesResponse(candidates=[]))
    provider = _azure({"attribute_candidates": responses})
    agent = AttributeExtractionAgent(cfg, provider)
    state = {
        "document": doc,
        "chunks": chunks,
        "stages": {"chunking": StageRecord(stage="chunking", status="succeeded", attempts=1)},
    }
    delta = agent.run(state)  # would raise StageFailedError if ungrounded evidence leaked through

    everything = [*delta["attributes"], *delta["discarded_attributes"]]
    names = {a.display_name for a in everything}
    assert "Activation Fee" not in names
    lead = [a for a in everything if a.display_name == "Provisioning Lead Time"]
    assert lead, "grounded LLM candidate must be scored (accepted, reviewed or discarded)"
    assert all(ev.text in target.source_text for a in lead for ev in a.evidence)
    trace = delta["traces"][-1]
    assert trace.data.get("llm_candidate_quote_dropped") == 1
    assert trace.data.get("llm_candidate_sentence_dropped") == 1


def test_attribute_extraction_merges_candidates_that_share_an_id(cfg, telecom_spec_md) -> None:
    """The LLM repeating a rules candidate (same chunk, name and evidence) must not produce
    two records with the same attribute id — that failed output validation with
    'duplicate attribute id' after every retry."""
    from doc_extractor.chunking import chunk_document
    from doc_extractor.ingest import load_document

    doc = load_document(telecom_spec_md, cfg)
    chunks = chunk_document(doc, cfg)
    target = next(c for c in chunks if "5 working days" in c.source_text)
    duplicate = tasks.CandidateProposal(
        raw_name="Provisioning Lead Time", source_text_quote="5 working days", sentence=None
    )
    responses = [
        # the same proposal twice in one chunk -> identical attribute id
        tasks.AttributeCandidatesResponse(candidates=[duplicate, duplicate])
        if chunk is target
        else tasks.AttributeCandidatesResponse(candidates=[])
        for chunk in chunks
    ]
    state = {
        "document": doc,
        "chunks": chunks,
        "stages": {"chunking": StageRecord(stage="chunking", status="succeeded", attempts=1)},
    }

    delta = AttributeExtractionAgent(cfg, _azure({"attribute_candidates": responses})).run(state)

    everything = [*delta["attributes"], *delta["discarded_attributes"]]
    all_ids = [a.attribute_id for a in everything]
    assert len(all_ids) == len(set(all_ids))
    assert sum(a.display_name == "Provisioning Lead Time" for a in everything) == 1
    assert delta["traces"][-1].data.get("candidates_same_id_merged") == 1


# --------------------------------------------------------------------------------------
# Agent 8: binding judgement for attributes the rules pass left unmapped
# --------------------------------------------------------------------------------------
def _state_through_reviewer(cfg) -> dict[str, Any]:
    provider = ScriptedProvider({}, name="rules")
    state = base_state(StageName.attribute_storage)
    for stage, agent_cls in [
        (StageName.entity_generation, EntityGenerationAgent),
        (StageName.entity_normalization, EntityNormalizationAgent),
        (StageName.entity_reviewer, EntityReviewerAgent),
    ]:
        state.update(agent_cls(cfg, provider).run(state))
        state["stages"][str(stage)] = StageRecord(stage=stage, attempts=1, status="succeeded")
    return state


def test_attribute_mapping_binds_leftover_attributes_via_azure(cfg) -> None:
    from doc_extractor.agents.attribute_mapping import AttributeMappingAgent

    state = _state_through_reviewer(cfg)
    # Only the dangling "Provisioning Time" attribute survives the rules pass unmapped.
    response = tasks.BindingJudgementResponse(
        bindings=[
            tasks.BindingJudgement(
                attribute_id="attr-provisioningtime",
                entity_name="service",  # case-insensitive match on the canonical name
                evidence_quote="Provisioning of the service",
                reason="the sentence describes provisioning of the service",
            ),
            # unknown attribute id -> ignored
            tasks.BindingJudgement(attribute_id="attr-hallucinated", entity_name="Service", reason="made up"),
        ]
    )
    provider = _azure({"binding_judgement": [response]})
    delta = AttributeMappingAgent(cfg, provider).run(state)

    [(task_name, payload)] = provider.call_log
    assert task_name == "binding_judgement"
    assert payload["chunk_id"] == CHUNK_SUBSCRIBER.chunk_id
    assert [a["attribute_id"] for a in payload["attributes"]] == ["attr-provisioningtime"]
    assert payload["candidate_entities"] == ["EVC", "Service", "Service Provider", "Subscriber", "UNI"]

    mapping = next(m for m in delta["mappings"] if m.attribute_id == "attr-provisioningtime")
    service = next(c for c in state["canonical_entities"] if c.canonical_name == "Service")
    assert mapping.entity_id == service.canonical_id
    assert mapping.confidence == pytest.approx(cfg.mapping.llm_binding_confidence)
    assert [e.text for e in mapping.supporting_evidence] == ["Provisioning of the service"]
    assert delta["unmapped_attribute_ids"] == []


@pytest.mark.parametrize(
    ("judgement", "expect_mapped", "expect_evidence"),
    [
        # entity not in the candidate list -> dropped
        (
            tasks.BindingJudgement(
                attribute_id="attr-provisioningtime", entity_name="Bank", reason="not listed"
            ),
            False,
            None,
        ),
        # null entity -> stays unmapped
        (tasks.BindingJudgement(attribute_id="attr-provisioningtime", reason="no owner"), False, None),
        # quote not in the chunk -> mapping kept at reduced confidence, without evidence
        (
            tasks.BindingJudgement(
                attribute_id="attr-provisioningtime",
                entity_name="Service",
                evidence_quote="text that is not in the chunk",
                reason="ungrounded quote",
            ),
            True,
            [],
        ),
    ],
)
def test_attribute_mapping_grounds_azure_binding_judgements(
    cfg, judgement, expect_mapped, expect_evidence
) -> None:
    from doc_extractor.agents.attribute_mapping import AttributeMappingAgent

    state = _state_through_reviewer(cfg)
    provider = _azure({"binding_judgement": [tasks.BindingJudgementResponse(bindings=[judgement])]})
    delta = AttributeMappingAgent(cfg, provider).run(state)

    mapped = {m.attribute_id: m for m in delta["mappings"]}
    assert ("attr-provisioningtime" in mapped) is expect_mapped
    if expect_mapped:
        mapping = mapped["attr-provisioningtime"]
        assert [e.text for e in mapping.supporting_evidence] == expect_evidence
        assert mapping.confidence == pytest.approx(0.8 * cfg.mapping.llm_binding_confidence)
    else:
        assert "attr-provisioningtime" in delta["unmapped_attribute_ids"]


def test_attribute_mapping_makes_no_azure_call_when_nothing_is_pending(cfg) -> None:
    from doc_extractor.agents.attribute_mapping import AttributeMappingAgent

    state = _state_through_reviewer(cfg)
    state["attributes"] = [a for a in state["attributes"] if a.attribute_id != "attr-provisioningtime"]
    provider = _azure({})
    AttributeMappingAgent(cfg, provider).run(state)
    assert provider.call_log == []
