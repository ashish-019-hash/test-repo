"""Deterministic, offline `LLMProvider`.

Every task answer here is derived only from the payload and the lexicon YAML
files — never guessed. This lets the whole pipeline run (and its tests pass)
without network access, and keeps behaviour identical whether or not an agent
happens to call the provider.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import yaml

from doc_extractor.config.models import AppConfig
from doc_extractor.llm.base import LLMCallResult, LLMTask
from doc_extractor.llm.tasks import (
    AttributeCandidatesResponse,
    BindingJudgement,
    BindingJudgementResponse,
    CandidateProposal,
    EntityProposal,
    EntityProposalsResponse,
    EntityQualityResponse,
)

_WORD_BOUNDARY = r"(?<![\w-]){}(?![\w-])"

_BINDING_VERBS = r"(?:has|have|shall have|must have)"


class RuleBasedProvider:
    """Zero-network `LLMProvider`; never fabricates an answer it cannot ground."""

    name = "rules"

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def call(self, task: LLMTask, payload: dict[str, Any]) -> LLMCallResult:
        handler = getattr(self, f"_{task.name}", None)
        if handler is None:
            raise ValueError(f"RuleBasedProvider has no handler for task {task.name!r}")
        response = handler(payload)
        return LLMCallResult(response=response, cache_hit=False, system_fingerprint=None)

    # -- attribute_candidates -------------------------------------------------------

    def _attribute_candidates(self, payload: dict[str, Any]) -> AttributeCandidatesResponse:
        return AttributeCandidatesResponse(candidates=self._generate_candidates(payload))

    def _generate_candidates(self, payload: dict[str, Any]) -> list[CandidateProposal]:
        try:
            from doc_extractor.scoring.candidates import generate_candidates
        except ImportError:
            return []

        try:
            sig = inspect.signature(generate_candidates)
        except (TypeError, ValueError):
            return []

        kwargs: dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            if pname in payload:
                kwargs[pname] = payload[pname]
            elif param.default is inspect.Parameter.empty and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return []  # a required field is missing from the payload

        try:
            raw_candidates = generate_candidates(**kwargs)
        except Exception:
            return []

        proposals: list[CandidateProposal] = []
        for c in raw_candidates:
            raw_name = getattr(c, "raw_name", None)
            source_text = getattr(c, "source_text", None)
            if not raw_name or not source_text:
                continue
            proposals.append(
                CandidateProposal(
                    raw_name=raw_name,
                    source_text_quote=source_text,
                    sentence=getattr(c, "sentence", None),
                )
            )
        return proposals

    # -- entity_proposals -------------------------------------------------------------

    def _entity_proposals(self, payload: dict[str, Any]) -> EntityProposalsResponse:
        source_text = payload.get("source_text") or ""
        chunk_id = payload.get("chunk_id") or ""
        entities: list[EntityProposal] = []
        for canonical_name, entity_type, patterns in self._known_entity_patterns():
            for pattern in patterns:
                match = pattern.search(source_text)
                if match is None:
                    continue
                entities.append(
                    EntityProposal(
                        entity_name=canonical_name,
                        entity_type=entity_type,
                        chunk_id=chunk_id,
                        evidence_quote=match.group(0),
                    )
                )
                break
        return EntityProposalsResponse(entities=entities)

    def _known_entity_patterns(self) -> list[tuple[str, str, list[re.Pattern[str]]]]:
        raw = self._load_known_entities()
        out: list[tuple[str, str, list[re.Pattern[str]]]] = []
        for name, spec in sorted(raw.items()):
            entity_type = str((spec or {}).get("type") or "unknown")
            names = [name, *((spec or {}).get("aliases") or [])]
            patterns = [
                re.compile(_WORD_BOUNDARY.format(re.escape(n)), re.IGNORECASE)
                for n in sorted(names, key=len, reverse=True)
            ]
            out.append((name, entity_type, patterns))
        return out

    def _load_known_entities(self) -> dict[str, Any]:
        path = self.cfg.lexicon_paths.get("known_entities")
        if path is None or not path.is_file():
            return {}
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}

    # -- entity_quality -----------------------------------------------------------

    def _entity_quality(self, payload: dict[str, Any]) -> EntityQualityResponse:
        return EntityQualityResponse(
            specificity_delta=0.0,
            real_world_delta=0.0,
            reason="rules provider: no adjustment",
        )

    # -- binding_judgement ----------------------------------------------------------

    def _binding_judgement(self, payload: dict[str, Any]) -> BindingJudgementResponse:
        candidate_entities = payload.get("candidate_entities") or []
        patterns = [
            (
                str(name),
                re.compile(
                    rf"(?:each|the|an?)\s+({re.escape(str(name))})\b.*?\b{_BINDING_VERBS}\b",
                    re.IGNORECASE,
                ),
            )
            for name in candidate_entities
        ]
        bindings: list[BindingJudgement] = []
        for attr in payload.get("attributes") or []:
            sentence = str(attr.get("sentence") or "")
            judgement = BindingJudgement(
                attribute_id=str(attr.get("attribute_id", "")),
                reason="rules provider: no match",
            )
            for name, pattern in patterns:
                if pattern.search(sentence) is not None:
                    judgement = BindingJudgement(
                        attribute_id=judgement.attribute_id,
                        entity_name=name,
                        evidence_quote=sentence,
                        reason=f"rules provider: matched binding pattern for {name!r}",
                    )
                    break
            bindings.append(judgement)
        return BindingJudgementResponse(bindings=bindings)


__all__ = ["RuleBasedProvider"]
