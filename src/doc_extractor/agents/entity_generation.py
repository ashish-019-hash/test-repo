"""Agent 5: EntityGenerationAgent.

Deterministic entity generation from four ordered sources (binding, promotion, lexicon,
optionally LLM proposals in Azure mode). Precision over recall: every rejection rule and
the confidence floor exist to keep noisy candidates out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from doc_extractor.agents.base import BaseAgent
from doc_extractor.exceptions import StageValidationError
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.reviewer import load_lexicons
from doc_extractor.schemas.attribute import Attribute
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.common import Evidence
from doc_extractor.schemas.entity import Entity
from doc_extractor.schemas.state import STAGE_ORDER, PipelineState, StageName
from doc_extractor.storage import ids

# Origin priority, highest first. Used when the same (name, type) is detected by more
# than one source: the highest-priority origin's label wins, but source_chunks /
# source_attributes / supporting_evidence are unioned across every contributing source.
_ORIGIN_PRIORITY: dict[str, int] = {"binding": 0, "promotion": 1, "lexicon": 2, "llm": 3}

_WORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _term_pattern(term: str) -> re.Pattern[str]:
    pat = _WORD_RE_CACHE.get(term)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        _WORD_RE_CACHE[term] = pat
    return pat


@dataclass
class _RawCandidate:
    name: str
    entity_type: str
    layer: str
    origin: str
    source_attributes: set[str] = field(default_factory=set)
    source_chunks: set[str] = field(default_factory=set)
    evidence: dict[tuple[str, str], Evidence] = field(default_factory=dict)

    def merge_key(self) -> tuple[str, str]:
        return (self.name.casefold(), self.entity_type)


def _type_layer(name: str, lexicons: Any) -> tuple[str, str]:
    ke = lexicons.known_entity_lookup(name)
    if ke is not None:
        return ke.type, ke.layer
    return "Object", "Other"


def _pick_evidence_for_chunk(attrs_for_chunk: list[Attribute], chunk: Chunk) -> Evidence | None:
    for attr in sorted(attrs_for_chunk, key=lambda a: a.attribute_id):
        candidate = attr.binding.evidence
        if candidate and candidate in chunk.source_text:
            return Evidence(chunk_id=chunk.chunk_id, text=candidate)
        if attr.source_text and attr.source_text in chunk.source_text:
            return Evidence(chunk_id=chunk.chunk_id, text=attr.source_text)
    return None


class EntityGenerationAgent(BaseAgent):
    name: ClassVar[StageName] = StageName.entity_generation
    requires: ClassVar[tuple[str, ...]] = ("attributes", "chunks")
    produces: ClassVar[tuple[str, ...]] = ("entities",)

    def validate_input(self, state: PipelineState) -> None:
        idx = STAGE_ORDER.index(self.name)
        prev = STAGE_ORDER[idx - 1]
        rec = (state.get("stages") or {}).get(str(prev))
        if rec is None or rec.status != "succeeded":
            raise StageValidationError(
                str(self.name),
                f"predecessor stage '{prev}' has status {rec.status if rec else 'missing'}",
                phase="input",
            )
        if "attributes" not in state:
            raise StageValidationError(
                str(self.name), "required state key 'attributes' is missing", phase="input"
            )
        chunks = state.get("chunks")
        if not chunks:
            raise StageValidationError(
                str(self.name), "required state key 'chunks' is missing or empty", phase="input"
            )

    def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
        document = state["document"]
        chunks = state["chunks"]
        attributes = list(state["attributes"])
        discarded = list(state.get("discarded_attributes", []))
        lexicons = load_lexicons(self.cfg)
        chunks_by_id = {c.chunk_id: c for c in chunks}

        raw: list[_RawCandidate] = []
        raw.extend(self._binding_candidates(attributes, chunks_by_id, lexicons))
        raw.extend(self._promotion_candidates(attributes, discarded, chunks_by_id, lexicons))
        raw.extend(self._lexicon_candidates(chunks, lexicons))
        if self.provider.name == "azure":
            raw.extend(self._llm_candidates(chunks, lexicons, trace))

        raw = self._apply_rejections(raw, attributes, lexicons)
        merged = self._merge(raw)

        origin_weights = self.cfg.entity_generation.origin_weights
        min_confidence = self.cfg.entity_generation.min_confidence
        entities: list[Entity] = []
        for cand in merged.values():
            mentions = len(cand.source_chunks) + len(cand.source_attributes)
            confidence = origin_weights.get(cand.origin, 0.5) * min(1.0, mentions / 3.0)
            if confidence < min_confidence:
                trace.count("entity_below_min_confidence")
                continue
            entities.append(
                Entity(
                    entity_id=ids.entity_id(document.document_id, cand.name, cand.entity_type),
                    entity_name=cand.name,
                    entity_type=cand.entity_type,
                    layer=cand.layer,  # type: ignore[arg-type]
                    source_attributes=sorted(cand.source_attributes),
                    source_chunks=sorted(cand.source_chunks),
                    supporting_evidence=sorted(cand.evidence.values(), key=lambda e: (e.chunk_id, e.text)),
                    origin=cand.origin,  # type: ignore[arg-type]
                    confidence=round(confidence, 6),
                )
            )
        entities.sort(key=lambda e: e.entity_id)
        trace.count("entities_generated", len(entities))
        return {"entities": entities}

    # ---- sources ------------------------------------------------------------------------
    def _binding_candidates(
        self, attributes: list[Attribute], chunks_by_id: dict[str, Chunk], lexicons: Any
    ) -> list[_RawCandidate]:
        by_name: dict[str, list[Attribute]] = {}
        for attr in attributes:
            if attr.entity:
                by_name.setdefault(attr.entity, []).append(attr)
        out: list[_RawCandidate] = []
        for name, attrs in by_name.items():
            etype, layer = _type_layer(name, lexicons)
            source_chunks = sorted({a.source_chunk for a in attrs})
            evidence: dict[tuple[str, str], Evidence] = {}
            for chunk_id in source_chunks:
                chunk = chunks_by_id.get(chunk_id)
                if chunk is None:
                    continue
                attrs_for_chunk = [a for a in attrs if a.source_chunk == chunk_id]
                ev = _pick_evidence_for_chunk(attrs_for_chunk, chunk)
                if ev is not None:
                    evidence[(ev.chunk_id, ev.text)] = ev
            out.append(
                _RawCandidate(
                    name=name,
                    entity_type=etype,
                    layer=layer,
                    origin="binding",
                    source_attributes={a.attribute_id for a in attrs},
                    source_chunks=set(source_chunks),
                    evidence=evidence,
                )
            )
        return out

    def _promotion_candidates(
        self,
        attributes: list[Attribute],
        discarded: list[Attribute],
        chunks_by_id: dict[str, Chunk],
        lexicons: Any,
    ) -> list[_RawCandidate]:
        by_name: dict[str, list[Attribute]] = {}
        for attr in attributes:
            if attr.entity:
                by_name.setdefault(attr.entity, []).append(attr)
        out: list[_RawCandidate] = []
        for rec in discarded:
            if rec.score.discard_reason != "promoted":
                continue
            name = rec.display_name
            etype, layer = _type_layer(name, lexicons)
            own_chunk = chunks_by_id.get(rec.source_chunk)
            evidence: dict[tuple[str, str], Evidence] = {}
            if own_chunk is not None and rec.source_text in own_chunk.source_text:
                ev = Evidence(chunk_id=rec.source_chunk, text=rec.source_text)
                evidence[(ev.chunk_id, ev.text)] = ev
            bound_attrs = by_name.get(name, [])
            out.append(
                _RawCandidate(
                    name=name,
                    entity_type=etype,
                    layer=layer,
                    origin="promotion",
                    source_attributes=set(),
                    source_chunks={rec.source_chunk},
                    evidence=evidence,
                )
            )
            # bound_attrs are already covered by the binding source; nothing further to add here.
            del bound_attrs
        return out

    def _lexicon_candidates(self, chunks: list[Chunk], lexicons: Any) -> list[_RawCandidate]:
        terms_by_canonical: dict[str, list[str]] = {}
        for name, ke in lexicons.known_entities.items():
            terms_by_canonical[name] = [name, *ke.aliases]
        for name in lexicons.sid_abes:
            terms_by_canonical.setdefault(name, [name])

        out: list[_RawCandidate] = []
        for canonical, terms in terms_by_canonical.items():
            patterns = [_term_pattern(t) for t in terms]
            chunk_matches: dict[str, str] = {}
            for chunk in chunks:
                for pattern in patterns:
                    m = pattern.search(chunk.source_text)
                    if m:
                        chunk_matches[chunk.chunk_id] = m.group(0)
                        break
            if len(chunk_matches) < self.cfg.entity_generation.min_mentions:
                continue
            etype, layer = _type_layer(canonical, lexicons)
            evidence = {(cid, span): Evidence(chunk_id=cid, text=span) for cid, span in chunk_matches.items()}
            out.append(
                _RawCandidate(
                    name=canonical,
                    entity_type=etype,
                    layer=layer,
                    origin="lexicon",
                    source_attributes=set(),
                    source_chunks=set(chunk_matches),
                    evidence=evidence,
                )
            )
        return out

    def _llm_candidates(
        self, chunks: list[Chunk], lexicons: Any, trace: TraceCollector
    ) -> list[_RawCandidate]:
        from doc_extractor.llm import tasks as llm_tasks  # lazy: azure-mode only

        task = _resolve_llm_task(llm_tasks, "entity_proposals")
        out: list[_RawCandidate] = []
        known_names = sorted(lexicons.known_entities)
        for chunk in chunks:
            payload = {
                "chunk_id": chunk.chunk_id,
                "source_text": chunk.source_text,
                "heading_path": chunk.heading_path,
                "known_entities": known_names,
            }
            result = self.provider.call(task, payload)
            trace.record_llm(result.cache_hit)
            for proposal in getattr(result.response, "proposals", []):
                quote = getattr(proposal, "evidence_quote", "")
                if quote not in chunk.source_text:
                    trace.count("evidence_rejected")
                    continue
                name = getattr(proposal, "name", "")
                etype, layer = _type_layer(name, lexicons)
                etype = getattr(proposal, "type", None) or etype
                ev = Evidence(chunk_id=chunk.chunk_id, text=quote)
                out.append(
                    _RawCandidate(
                        name=name,
                        entity_type=etype,
                        layer=layer,
                        origin="llm",
                        source_attributes=set(),
                        source_chunks={chunk.chunk_id},
                        evidence={(ev.chunk_id, ev.text): ev},
                    )
                )
        return out

    # ---- rejection + merge ----------------------------------------------------------------
    def _apply_rejections(
        self, raw: list[_RawCandidate], attributes: list[Attribute], lexicons: Any
    ) -> list[_RawCandidate]:
        attr_name_set = {a.attribute_name.casefold() for a in attributes} | {
            a.display_name.casefold() for a in attributes
        }
        out = []
        for cand in raw:
            name_cf = cand.name.casefold()
            if len(cand.name.strip()) < 2:
                continue
            if name_cf in lexicons.generic_terms:
                continue
            if name_cf in attr_name_set:
                continue
            out.append(cand)
        return out

    def _merge(self, raw: list[_RawCandidate]) -> dict[tuple[str, str], _RawCandidate]:
        merged: dict[tuple[str, str], _RawCandidate] = {}
        for cand in raw:
            key = cand.merge_key()
            existing = merged.get(key)
            if existing is None:
                merged[key] = _RawCandidate(
                    name=cand.name,
                    entity_type=cand.entity_type,
                    layer=cand.layer,
                    origin=cand.origin,
                    source_attributes=set(cand.source_attributes),
                    source_chunks=set(cand.source_chunks),
                    evidence=dict(cand.evidence),
                )
                continue
            existing.source_attributes |= cand.source_attributes
            existing.source_chunks |= cand.source_chunks
            existing.evidence.update(cand.evidence)
            if _ORIGIN_PRIORITY[cand.origin] < _ORIGIN_PRIORITY[existing.origin]:
                existing.origin = cand.origin
                existing.name = cand.name
                existing.layer = cand.layer
        return merged

    # ---- validation -------------------------------------------------------------------
    def validate_output(self, delta: dict[str, Any], state: PipelineState) -> None:
        super().validate_output(delta, state)
        entities: list[Entity] = delta["entities"]
        attr_ids = {a.attribute_id for a in state["attributes"]}
        chunk_ids = {c.chunk_id for c in state["chunks"]}
        chunks_by_id = {c.chunk_id: c for c in state["chunks"]}
        seen_ids: set[str] = set()
        for entity in entities:
            if entity.entity_id in seen_ids:
                raise StageValidationError(str(self.name), f"duplicate entity_id {entity.entity_id}")
            seen_ids.add(entity.entity_id)
            if not entity.supporting_evidence:
                raise StageValidationError(str(self.name), f"entity {entity.entity_id} has no evidence")
            if not entity.source_chunks:
                raise StageValidationError(str(self.name), f"entity {entity.entity_id} has no source chunk")
            for attr_id in entity.source_attributes:
                if attr_id not in attr_ids:
                    raise StageValidationError(
                        str(self.name), f"entity {entity.entity_id} references unknown attribute {attr_id}"
                    )
            for chunk_id in entity.source_chunks:
                if chunk_id not in chunk_ids:
                    raise StageValidationError(
                        str(self.name), f"entity {entity.entity_id} references unknown chunk {chunk_id}"
                    )
            for ev in entity.supporting_evidence:
                chunk = chunks_by_id.get(ev.chunk_id)
                if chunk is None or ev.text not in chunk.source_text:
                    raise StageValidationError(
                        str(self.name),
                        f"entity {entity.entity_id} evidence not grounded in chunk {ev.chunk_id}",
                    )


def _resolve_llm_task(module: Any, name: str) -> Any:
    const_name = name.upper()
    if hasattr(module, const_name):
        return getattr(module, const_name)
    if hasattr(module, "TASKS"):
        return module.TASKS[name]
    raise AttributeError(f"LLM task '{name}' not found in doc_extractor.llm.tasks")
