"""Format detection and the single ingest entrypoint used by the rest of the pipeline."""

from __future__ import annotations

from pathlib import Path

from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import IngestError
from doc_extractor.ingest import docx as _docx
from doc_extractor.ingest import pdf as _pdf
from doc_extractor.ingest import text as _text
from doc_extractor.ingest.base import Loader
from doc_extractor.schemas.document import Document, DocumentFormat
from doc_extractor.storage import ids

_EXTENSION_MAP: dict[str, DocumentFormat] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
}

_LOADERS: dict[DocumentFormat, Loader] = {
    "pdf": _pdf.LOADER,
    "docx": _docx.LOADER,
    "txt": _text.LOADER,
    "md": _text.LOADER,
}


def detect_format(path: str | Path) -> DocumentFormat:
    """Extension first, then magic bytes; unrecognized binary content falls back to text."""
    p = Path(path)
    fmt = _EXTENSION_MAP.get(p.suffix.lower())
    if fmt is not None:
        return fmt
    try:
        head = p.open("rb").read(8)
    except OSError as exc:
        raise IngestError(f"Cannot read {p}: {exc}") from exc
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return "docx"
    return "txt"


def load_document(path: str | Path, cfg: AppConfig) -> Document:
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"Input file not found: {p}")
    document_id = ids.document_id(p.read_bytes())
    fmt = detect_format(p)
    return _LOADERS[fmt].load(p, cfg, document_id)


__all__ = ["detect_format", "load_document"]
