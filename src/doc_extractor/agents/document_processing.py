"""Agent 1: DocumentProcessingAgent.

Loads the input file into a `Document` (blocks + pages), doing no
interpretation of content: no entity/attribute extraction happens here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from doc_extractor.agents.base import BaseAgent
from doc_extractor.exceptions import StageValidationError
from doc_extractor.ingest import load_document
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.schemas.state import PipelineState, StageName
from doc_extractor.storage import ids


class DocumentProcessingAgent(BaseAgent):
    name: ClassVar[StageName] = StageName.document_processing
    requires: ClassVar[tuple[str, ...]] = ("input_path",)
    produces: ClassVar[tuple[str, ...]] = ("document",)

    def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
        input_path = Path(state["input_path"])
        document = load_document(input_path, self.cfg)
        trace.set("format", document.format)
        trace.count("pages", len(document.pages))
        trace.count("blocks", len(document.blocks))
        trace.set("needs_ocr", document.needs_ocr)
        return {"document": document}

    def validate_output(self, delta: dict[str, Any], state: PipelineState) -> None:
        super().validate_output(delta, state)
        document = delta["document"]
        block_ids = {b.block_id for b in document.blocks}
        for page in document.pages:
            for block_id in page.block_ids:
                if block_id not in block_ids:
                    raise StageValidationError(
                        str(self.name),
                        f"page {page.page_number} references unknown block '{block_id}'",
                    )
        expected_id = ids.document_id(Path(state["input_path"]).read_bytes())
        if document.document_id != expected_id:
            raise StageValidationError(
                str(self.name),
                f"document_id '{document.document_id}' does not match file hash '{expected_id}'",
            )
