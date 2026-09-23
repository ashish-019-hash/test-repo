"""Deterministic, content-derived identifiers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def _h(*parts: str, n: int = 16) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:n]


def document_id(file_bytes: bytes) -> str:
    return "doc-" + hashlib.sha256(file_bytes).hexdigest()[:16]


def block_id(doc_id: str, page: int, n: int) -> str:
    return f"blk-{doc_id}-p{page}-{n:04d}"


def chunk_id(doc_id: str, page: int, n: int) -> str:
    return f"chunk-{doc_id}-p{page}-{n:04d}"


def attribute_id(doc_id: str, chunk: str, attribute_name: str, source_text: str) -> str:
    return "attr-" + _h(doc_id, chunk, attribute_name, source_text)


def entity_id(doc_id: str, entity_name: str, entity_type: str) -> str:
    return "ent-" + _h(doc_id, entity_name, entity_type)


def canonical_id(member_ids: Iterable[str]) -> str:
    return "cent-" + _h(*sorted(member_ids))


def mapping_id(attr_id: str, cent_id: str, relationship_type: str) -> str:
    return "map-" + _h(attr_id, cent_id, relationship_type)


def review_item_id(kind: str, ref_ids: Iterable[str]) -> str:
    return "hri-" + _h(kind, *sorted(ref_ids))


def config_hash(canonical_config_json: str, lexicon_contents: Iterable[str]) -> str:
    return _h(canonical_config_json, *lexicon_contents, n=16)
