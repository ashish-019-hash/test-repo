"""Output schema of the Document Processing Agent."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from doc_extractor.schemas.common import StrictModel

BlockKind = Literal["heading", "paragraph", "table", "list_item", "form_field", "furniture"]
DocumentFormat = Literal["pdf", "docx", "txt", "md"]


class Table(StrictModel):
    caption: str | None = None
    header: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    subject_hint: str | None = Field(
        default=None,
        description="Declared subject of the table (caption, spanning title row, or heading directly above).",
    )


class Block(StrictModel):
    block_id: str = Field(description='Format "blk-<doc>-p<page>-<nnnn>"')
    page: int
    kind: BlockKind
    text: str
    heading_level: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    table: Table | None = None


class Page(StrictModel):
    page_number: int
    text: str
    ocr_required: bool = False
    block_ids: list[str] = Field(default_factory=list)


class Document(StrictModel):
    document_id: str = Field(description='Format "doc-<sha256(file bytes)[:16]>"')
    file_name: str
    format: DocumentFormat
    title: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    pages: list[Page] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    needs_ocr: bool = False
    text_coverage: float = Field(default=1.0, ge=0.0, le=1.0)

    def block_map(self) -> dict[str, Block]:
        return {b.block_id: b for b in self.blocks}


class DocumentSummary(StrictModel):
    """Document header for final.json: ids and metadata only, no full text."""

    document_id: str
    file_name: str
    format: DocumentFormat
    title: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    page_count: int
    ocr_required_pages: list[int] = Field(default_factory=list)
    needs_ocr: bool = False
    text_coverage: float
    block_count: int
    chunk_count: int

    @classmethod
    def from_document(cls, doc: Document, chunk_count: int) -> DocumentSummary:
        return cls(
            document_id=doc.document_id,
            file_name=doc.file_name,
            format=doc.format,
            title=doc.title,
            metadata=doc.metadata,
            page_count=len(doc.pages),
            ocr_required_pages=sorted(p.page_number for p in doc.pages if p.ocr_required),
            needs_ocr=doc.needs_ocr,
            text_coverage=doc.text_coverage,
            block_count=len(doc.blocks),
            chunk_count=chunk_count,
        )
