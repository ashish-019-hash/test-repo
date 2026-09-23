"""Agent 2: ChunkingAgent.

Slices/joins the document's blocks into `Chunk`s. Never rewrites text: every
chunk's `source_text` must equal the newline-join of its blocks' text.
"""

from __future__ import annotations

from typing import Any, ClassVar

from doc_extractor.agents.base import BaseAgent
from doc_extractor.chunking import chunk_document
from doc_extractor.exceptions import StageValidationError
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.schemas.state import PipelineState, StageName


class ChunkingAgent(BaseAgent):
    name: ClassVar[StageName] = StageName.chunking
    requires: ClassVar[tuple[str, ...]] = ("document",)
    produces: ClassVar[tuple[str, ...]] = ("chunks",)

    def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
        document = state["document"]
        chunks = chunk_document(document, self.cfg)
        trace.set("strategy", self.cfg.chunking.strategy)
        trace.count("chunks", len(chunks))
        return {"chunks": chunks}

    def validate_output(self, delta: dict[str, Any], state: PipelineState) -> None:
        super().validate_output(delta, state)
        document = state["document"]
        chunks = delta["chunks"]
        chunk_ids = [c.chunk_id for c in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise StageValidationError(str(self.name), "chunk ids are not unique")
        if chunk_ids != sorted(chunk_ids):
            raise StageValidationError(str(self.name), "chunks are not sorted by chunk_id")

        block_map = document.block_map()
        covered: set[str] = set()
        for chunk in chunks:
            if not chunk.block_ids:
                raise StageValidationError(str(self.name), f"chunk {chunk.chunk_id} has no blocks")
            expected_text = "\n".join(block_map[bid].text for bid in chunk.block_ids)
            if expected_text != chunk.source_text:
                raise StageValidationError(
                    str(self.name), f"chunk {chunk.chunk_id} source_text does not match its blocks"
                )
            for bid in chunk.block_ids:
                if bid not in block_map:
                    raise StageValidationError(
                        str(self.name), f"chunk {chunk.chunk_id} references unknown block '{bid}'"
                    )
            covered.update(chunk.block_ids)

        missing = {b.block_id for b in document.blocks} - covered
        if missing:
            raise StageValidationError(
                str(self.name), f"blocks not covered by any chunk: {sorted(missing)[:5]}"
            )
