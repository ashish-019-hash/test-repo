"""Chunking entrypoint: dispatches to the configured strategy."""

from __future__ import annotations

from doc_extractor.chunking import strategies
from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.document import Document

_STRATEGIES = {
    "by_page": strategies.by_page,
    "fixed_tokens": strategies.fixed_tokens,
    "heading_aware": strategies.heading_aware,
}


def chunk_document(document: Document, cfg: AppConfig) -> list[Chunk]:
    strategy = _STRATEGIES[cfg.chunking.strategy]
    return strategy(document.blocks, document.document_id, cfg)


__all__ = ["chunk_document"]
