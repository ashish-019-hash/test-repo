"""Agent 3: AttributeExtractionAgent.

Deterministic pipeline per document: generate candidates -> bind every candidate to an
entity -> score every candidate (negative.has_sub_attributes needs the *global* count of
how many other candidates bind to a given name, so binding runs as a first pass over every
chunk before scoring runs as a second pass) -> route accept/review/discard -> dedup
(entity, attribute_name) within the accepted+review set.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

from doc_extractor.agents.base import BaseAgent
from doc_extractor.exceptions import StageValidationError
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.schemas.attribute import Attribute, AttributeCandidate, EntityBinding
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.common import Evidence, Provenance
from doc_extractor.schemas.document import Document
from doc_extractor.schemas.state import HumanReviewItem, PipelineState, StageName
from doc_extractor.scoring.binding import bind_entity, known_entities_for_document, table_subject_entities
from doc_extractor.scoring.candidates import generate_candidates
from doc_extractor.scoring.engine import score as score_candidate
from doc_extractor.scoring.lexicons import Lexicons, load_lexicons
from doc_extractor.scoring.normalize import (
    aliases_for,
    attribute_type_for,
    camel_case,
    optionality_for,
    unit_for,
    value_domain_for,
)
from doc_extractor.scoring.signals import ScoringContext
from doc_extractor.storage import ids


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _row_evidence_text(source_text: str, raw_name: str) -> str:
    """The single spec-table row / kv line naming `raw_name`, else the whole text."""
    prefix = f"| {raw_name} |"
    for line in source_text.split("\n"):
        if line.startswith(prefix):
            return line
    return source_text


def _evidence_text_for(candidate: AttributeCandidate) -> str:
    if candidate.origin in ("spec_table", "kv_pair"):
        return _row_evidence_text(candidate.source_text, candidate.raw_name)
    if candidate.sentence:
        return candidate.sentence
    return candidate.source_text


class AttributeExtractionAgent(BaseAgent):
    name: ClassVar[StageName] = StageName.attribute_extraction
    requires: ClassVar[tuple[str, ...]] = ("chunks", "document")
    produces: ClassVar[tuple[str, ...]] = ("attributes", "discarded_attributes")

    def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
        document: Document = state["document"]
        chunks: list[Chunk] = list(state["chunks"])
        lexicons = load_lexicons(self.cfg)
        block_map = document.block_map()

        known_entities = known_entities_for_document(document, lexicons)
        table_subjects = table_subject_entities(document, known_entities, lexicons)

        records: list[tuple[Chunk, AttributeCandidate]] = []
        for chunk in chunks:
            records.extend((chunk, c) for c in generate_candidates(chunk, document, lexicons, self.cfg))
        trace.count("candidates_generated", len(records))

        if self.provider.name == "azure":
            records.extend(self._llm_candidates(chunks, trace))
            trace.count("candidates_after_llm", len(records))

        all_candidates = [c for _, c in records]

        def make_ctx(chunk: Chunk, cand: AttributeCandidate, bind_counts: dict[str, int]) -> ScoringContext:
            block = block_map.get(cand.block_id)
            table = block.table if block is not None else None
            return ScoringContext(
                candidate=cand,
                chunk=chunk,
                block=block,
                heading_path=list(chunk.heading_path),
                table=table,
                sentence=cand.sentence,
                document_title=document.title,
                all_candidates=all_candidates,
                known_entities=known_entities,
                table_subjects=table_subjects,
                bind_counts=bind_counts,
            )

        # Pass 1: bind every candidate (binding does not depend on bind_counts).
        bindings: dict[int, EntityBinding] = {}
        for chunk, cand in records:
            ctx = make_ctx(chunk, cand, {})
            bindings[id(cand)] = bind_entity(ctx, lexicons, self.cfg)

        bind_counts: dict[str, int] = {}
        for _chunk, cand in records:
            binding = bindings[id(cand)]
            if binding.entity_name and _norm(binding.entity_name) != _norm(cand.raw_name):
                key = _norm(binding.entity_name)
                bind_counts[key] = bind_counts.get(key, 0) + 1

        # Pass 2: score every candidate using the final bind_counts, and build records.
        accepted_or_review: list[Attribute] = []
        discarded: list[Attribute] = []
        human_review_queue: list[HumanReviewItem] = []

        for chunk, cand in records:
            binding = bindings[id(cand)]
            ctx = make_ctx(chunk, cand, bind_counts)
            breakdown = score_candidate(ctx, lexicons, self.cfg)
            attr = self._build_attribute(
                document, chunk, block_map, cand, binding, breakdown, lexicons, known_entities
            )

            if breakdown.route == "discard":
                discarded.append(attr)
                continue
            accepted_or_review.append(attr)
            if breakdown.route == "review":
                human_review_queue.append(
                    HumanReviewItem(
                        item_id=ids.review_item_id("attribute_review", [attr.attribute_id]),
                        kind="attribute_review",
                        ref_ids=[attr.attribute_id],
                        score=breakdown.clamped,
                        reason=f'confidence {breakdown.clamped:.2f} in review band for "{attr.display_name}"',
                        payload={"attribute_id": attr.attribute_id, "raw_name": attr.display_name},
                    )
                )

        attributes = self._dedup(accepted_or_review)
        attributes.sort(key=lambda a: a.attribute_id)
        discarded.sort(key=lambda a: a.attribute_id)
        # Review items must only reference attributes that survived in-document dedup.
        surviving = {a.attribute_id for a in attributes}
        human_review_queue = [h for h in human_review_queue if set(h.ref_ids) <= surviving]
        human_review_queue.sort(key=lambda h: h.item_id)

        trace.count("attributes_accepted", len(attributes))
        trace.count("attributes_discarded", len(discarded))
        trace.count("attribute_reviews", len(human_review_queue))

        return {
            "attributes": attributes,
            "discarded_attributes": discarded,
            "human_review_queue": human_review_queue,
        }

    # ---- helpers --------------------------------------------------------------------------
    def _llm_candidates(
        self, chunks: list[Chunk], trace: TraceCollector
    ) -> list[tuple[Chunk, AttributeCandidate]]:
        """Azure mode only: propose extra candidates, dropping any quote that isn't a
        literal substring of its chunk's source text."""
        from doc_extractor.llm import tasks as llm_tasks  # lazy: only imported in azure mode

        out: list[tuple[Chunk, AttributeCandidate]] = []
        for chunk in chunks:
            result = self.provider.call(
                llm_tasks.ATTRIBUTE_CANDIDATES_TASK,
                {"chunk_id": chunk.chunk_id, "source_text": chunk.source_text},
            )
            response = cast(llm_tasks.AttributeCandidatesResponse, result.response)
            for proposal in response.candidates:
                if proposal.source_text_quote not in chunk.source_text:
                    trace.count("llm_candidate_quote_dropped")
                    continue
                out.append(
                    (
                        chunk,
                        AttributeCandidate(
                            raw_name=proposal.raw_name,
                            chunk_id=chunk.chunk_id,
                            block_id=chunk.block_ids[0] if chunk.block_ids else "",
                            source_text=chunk.source_text,
                            origin="llm",
                            sentence=proposal.sentence,
                        ),
                    )
                )
        return out

    def _build_attribute(
        self,
        document: Document,
        chunk: Chunk,
        block_map: dict[str, Any],
        cand: AttributeCandidate,
        binding: EntityBinding,
        breakdown: Any,
        lexicons: Lexicons,
        known_entities: dict[str, Any],
    ) -> Attribute:
        cells = cand.cells
        description = cells.get("description")
        attribute_name = camel_case(cand.raw_name)
        evidence_text = _evidence_text_for(cand)
        entity_name = binding.entity_name
        layer = None
        if entity_name is not None:
            ke = known_entities.get(entity_name) or lexicons.known_entity_lookup(entity_name)
            layer = getattr(ke, "layer", None)
        block = block_map.get(cand.block_id)
        provenance = Provenance(
            file=document.file_name,
            page=block.page if block is not None else 0,
            block_id=cand.block_id,
            bbox=block.bbox if block is not None else None,
        )
        source_text = evidence_text
        attribute_id = ids.attribute_id(document.document_id, cand.chunk_id, attribute_name, source_text)
        return Attribute(
            attribute_id=attribute_id,
            attribute_name=attribute_name,
            display_name=cand.raw_name,
            aliases=aliases_for(cand.raw_name, description, lexicons),
            attribute_type=attribute_type_for(cells, lexicons, cand.raw_name),
            entity=entity_name,
            binding=binding,
            layer=layer,
            unit=unit_for(cells.get("units")),
            cardinality=None,
            optionality=optionality_for(cells),
            value_domain=value_domain_for(cells),
            default=cells.get("default"),
            description=description,
            source_document=document.document_id,
            source_chunk=cand.chunk_id,
            source_text=source_text,
            provenance=provenance,
            evidence=[Evidence(chunk_id=cand.chunk_id, text=evidence_text)],
            confidence=breakdown.clamped,
            score=breakdown,
            origin=cand.origin,
        )

    def _dedup(self, attributes: list[Attribute]) -> list[Attribute]:
        best: dict[tuple[str | None, str], Attribute] = {}
        for attr in attributes:
            key = (attr.entity, attr.attribute_name)
            existing = best.get(key)
            if existing is None:
                best[key] = attr
                continue
            # Higher confidence wins; ties break on attribute_id so the result is order-independent.
            if (attr.confidence, attr.attribute_id) > (existing.confidence, existing.attribute_id):
                keep, other = attr, existing
            else:
                keep, other = existing, attr
            merged_evidence = list(keep.evidence)
            for ev in other.evidence:
                if ev not in merged_evidence:
                    merged_evidence.append(ev)
            merged_aliases = list(keep.aliases)
            if other.display_name != keep.display_name and other.display_name not in merged_aliases:
                merged_aliases.append(other.display_name)
            for alias in other.aliases:
                if alias not in merged_aliases:
                    merged_aliases.append(alias)
            best[key] = keep.model_copy(update={"evidence": merged_evidence, "aliases": merged_aliases})
        return list(best.values())

    def validate_output(self, delta: dict[str, Any], state: PipelineState) -> None:
        super().validate_output(delta, state)
        chunks: list[Chunk] = list(state["chunks"])
        chunk_by_id = {c.chunk_id: c for c in chunks}
        seen_ids: set[str] = set()
        for attr in [*delta["attributes"], *delta["discarded_attributes"]]:
            if attr.source_chunk not in chunk_by_id:
                raise StageValidationError(
                    str(self.name),
                    f'attribute "{attr.attribute_id}" references unknown chunk "{attr.source_chunk}"',
                )
            for ev in attr.evidence:
                ev_chunk = chunk_by_id.get(ev.chunk_id)
                if ev_chunk is None or ev.text not in ev_chunk.source_text:
                    raise StageValidationError(
                        str(self.name),
                        f'attribute "{attr.attribute_id}" evidence is not a substring of chunk "{ev.chunk_id}"',
                    )
            if attr.attribute_id in seen_ids:
                raise StageValidationError(str(self.name), f'duplicate attribute id "{attr.attribute_id}"')
            seen_ids.add(attr.attribute_id)
            if abs(attr.confidence - attr.score.clamped) > 1e-9:
                raise StageValidationError(
                    str(self.name), f'attribute "{attr.attribute_id}" confidence does not equal score.clamped'
                )


__all__ = ["AttributeExtractionAgent"]
