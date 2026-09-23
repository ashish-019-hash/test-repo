"""Output schema of the Chunking Agent."""

from __future__ import annotations

from pydantic import Field

from doc_extractor.schemas.common import StrictModel


class Chunk(StrictModel):
    chunk_id: str = Field(description='Format "chunk-<doc>-p<page>-<nnnn>"')
    document_id: str
    page_number: int
    page_end: int
    source_text: str
    block_ids: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    token_count: int = 0
    strategy: str = "heading_aware"
