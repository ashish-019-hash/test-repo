"""Agent 9: FinalOutputAgent.

Assembles the immutable `FinalOutput` payload and writes every deterministic output
file (`final.json`, `entities.json`, `mappings.json`, `review_queue.json`, and optionally
`graph.json`/`graph.graphml`). Modifies nothing: every record is copied verbatim from
upstream state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from doc_extractor.agents.base import BaseAgent
from doc_extractor.exceptions import StageValidationError
from doc_extractor.graph.export import export_graph
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.schemas.attribute import Attribute
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.document import Document, DocumentSummary
from doc_extractor.schemas.entity import CanonicalEntity, Entity
from doc_extractor.schemas.mapping import Mapping
from doc_extractor.schemas.output import FinalOutput, ReviewSection, TraceEdge
from doc_extractor.schemas.review import DuplicateGroup, ReviewDecision
from doc_extractor.schemas.state import HumanReviewItem, PipelineState, StageName
from doc_extractor.storage.canonical_json import dumps
from doc_extractor.storage.json_store import write_models


class FinalOutputAgent(BaseAgent):
    name: ClassVar[StageName] = StageName.final_output
    requires: ClassVar[tuple[str, ...]] = ("document", "chunks", "canonical_entities", "review_decisions")
    produces: ClassVar[tuple[str, ...]] = ("final_output",)

    def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
        document: Document = state["document"]
        chunks: list[Chunk] = state["chunks"]
        attributes: list[Attribute] = sorted(state.get("attributes", []), key=lambda a: a.attribute_id)
        entities: list[Entity] = state.get("entities", [])
        canonical_entities: list[CanonicalEntity] = state["canonical_entities"]
        mappings: list[Mapping] = state.get("mappings", [])
        decisions: list[ReviewDecision] = state["review_decisions"]
        duplicate_groups: list[DuplicateGroup] = state.get("duplicate_groups", [])
        unmapped_ids: list[str] = state.get("unmapped_attribute_ids", [])
        review_queue: list[HumanReviewItem] = state.get("human_review_queue", [])

        doc_summary = DocumentSummary.from_document(document, len(chunks))

        dangling_ids = sorted(a.attribute_id for a in attributes if a.binding.scope == "dangling")

        review = ReviewSection(
            duplicates_removed=sorted(
                (g for g in duplicate_groups if len(g.member_ids) > 1), key=lambda g: g.canonical_id
            ),
            entities_requiring_review=sorted(
                (d for d in decisions if d.validation_status == "REVIEW"), key=lambda d: d.entity_id
            ),
            rejected_entities=sorted(
                (d for d in decisions if d.validation_status == "REJECT"), key=lambda d: d.entity_id
            ),
        )

        trace_edges = self._build_trace(document, chunks, attributes, entities, canonical_entities, mappings)

        final = FinalOutput(
            document=doc_summary,
            attributes=attributes,
            entities=canonical_entities,
            mappings=mappings,
            review=review,
            unmapped_attribute_ids=sorted(unmapped_ids),
            dangling_attribute_ids=dangling_ids,
            trace=trace_edges,
        )

        out_dir = Path(state["out_dir"])
        write_models(out_dir, "final", final)
        write_models(out_dir, "entities", canonical_entities)
        write_models(out_dir, "mappings", mappings)
        write_models(out_dir, "review_queue", review_queue)
        if self.cfg.output.graph_export:
            ext = "graphml" if self.cfg.output.graph_export == "graphml" else "json"
            export_graph(final, out_dir / f"graph.{ext}", self.cfg.output.graph_export)

        trace.count("final_attributes", len(attributes))
        trace.count("final_entities", len(canonical_entities))
        trace.count("final_mappings", len(mappings))
        return {"final_output": final}

    @staticmethod
    def _build_trace(
        document: Document,
        chunks: list[Chunk],
        attributes: list[Attribute],
        entities: list[Entity],
        canonical_entities: list[CanonicalEntity],
        mappings: list[Mapping],
    ) -> list[TraceEdge]:
        edges: list[TraceEdge] = []
        for chunk in chunks:
            edges.append(TraceEdge(from_id=document.document_id, to_id=chunk.chunk_id, kind="document>chunk"))
        for attr in attributes:
            edges.append(
                TraceEdge(from_id=attr.source_chunk, to_id=attr.attribute_id, kind="chunk>attribute")
            )
        for entity in entities:
            for attr_id in entity.source_attributes:
                edges.append(TraceEdge(from_id=attr_id, to_id=entity.entity_id, kind="attribute>entity"))
        for cent in canonical_entities:
            for member_id in cent.member_ids:
                edges.append(TraceEdge(from_id=member_id, to_id=cent.canonical_id, kind="entity>canonical"))
        for mapping in mappings:
            edges.append(
                TraceEdge(from_id=mapping.attribute_id, to_id=mapping.mapping_id, kind="attribute>mapping")
            )
            edges.append(
                TraceEdge(from_id=mapping.mapping_id, to_id=mapping.entity_id, kind="mapping>entity")
            )
        edges.sort(key=lambda e: (e.kind, e.from_id, e.to_id))
        return edges

    # ---- validation -----------------------------------------------------------------------
    def validate_output(self, delta: dict[str, Any], state: PipelineState) -> None:
        super().validate_output(delta, state)
        final: FinalOutput = delta["final_output"]

        known_ids: set[str] = {final.document.document_id}
        known_ids |= {a.attribute_id for a in final.attributes}
        known_ids |= {c.canonical_id for c in final.entities}
        known_ids |= {m for c in final.entities for m in c.member_ids}
        known_ids |= {m.mapping_id for m in final.mappings}
        known_ids |= {c.chunk_id for c in state["chunks"]}

        for edge in final.trace:
            if edge.from_id not in known_ids:
                raise StageValidationError(
                    str(self.name), f"trace edge from_id {edge.from_id} not in payload"
                )
            if edge.to_id not in known_ids:
                raise StageValidationError(str(self.name), f"trace edge to_id {edge.to_id} not in payload")

        round_tripped = FinalOutput.model_validate(json.loads(dumps(final)))
        if dumps(round_tripped) != dumps(final):
            raise StageValidationError(
                str(self.name), "canonical_json round-trip of final_output is not equal"
            )
