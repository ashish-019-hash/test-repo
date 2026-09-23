"""generate_candidates(chunk, document, lexicons, cfg) -> list[AttributeCandidate].

Deterministic candidate generation: spec-table rows, 2-column kv tables, form fields, and
prose regex scans (known-entity mentions, gazetteer terms, RFC-2119 tails, nominalisations).
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.attribute import AttributeCandidate
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.document import Block, Document
from doc_extractor.scoring.binding import known_entities_for_document
from doc_extractor.scoring.lexicons import KnownEntity, Lexicons
from doc_extractor.scoring.signals import merged_span_for, split_sentences
from doc_extractor.scoring.syntactic import compile_patterns, split_rfc2119_tail

_NOMINALISATION_SUFFIX_RE = re.compile(r"(?:tion|ing|ment|ance)$", re.IGNORECASE)
_SENTENCE_INITIAL_WORD_RE = re.compile(r"^([A-Z][a-z]+)\b")


def _header_categories(header: list[str], cfg: AppConfig) -> dict[int, str]:
    alias_lists = cfg.scoring.spec_table_headers
    result: dict[int, str] = {}
    for idx, cell in enumerate(header):
        norm_cell = cell.strip().lower()
        for category, names in alias_lists.items():
            if norm_cell in (n.lower() for n in names):
                result[idx] = category
                break
    return result


def _spec_table_candidates(block: Block, chunk: Chunk, cfg: AppConfig) -> list[AttributeCandidate]:
    table = block.table
    if table is None or not table.header or not table.rows:
        return []
    categories = _header_categories(table.header, cfg)
    name_idx = next((i for i, c in categories.items() if c == "name"), None)
    if name_idx is None:
        return []
    out: list[AttributeCandidate] = []
    for row in table.rows:
        if name_idx >= len(row) or not row[name_idx].strip():
            continue
        cells = {c: row[i].strip() for i, c in categories.items() if i < len(row) and row[i].strip()}
        out.append(
            AttributeCandidate(
                raw_name=row[name_idx].strip(),
                chunk_id=chunk.chunk_id,
                block_id=block.block_id,
                source_text=chunk.source_text,
                origin="spec_table",
                cells=cells,
            )
        )
    return out


def _kv_table_candidates(block: Block, chunk: Chunk, cfg: AppConfig) -> list[AttributeCandidate]:
    table = block.table
    if table is None or len(table.header) != 2 or not table.rows:
        return []
    if _header_categories(table.header, cfg):
        return []  # already a recognized spec table (handled above)
    out: list[AttributeCandidate] = []
    for row in table.rows:
        if len(row) < 2 or not row[0].strip():
            continue
        cells = {"description": row[1].strip()} if row[1].strip() else {}
        out.append(
            AttributeCandidate(
                raw_name=row[0].strip(),
                chunk_id=chunk.chunk_id,
                block_id=block.block_id,
                source_text=chunk.source_text,
                origin="kv_pair",
                cells=cells,
            )
        )
    return out


_FORM_FIELD_RE = re.compile(r"^(?P<label>[^:]{1,80}):\s*(?P<value>.*)$")


def _form_field_candidates(block: Block, chunk: Chunk) -> list[AttributeCandidate]:
    if block.kind != "form_field" or not block.text:
        return []
    m = _FORM_FIELD_RE.match(block.text.strip())
    if not m:
        return []
    label = m.group("label").strip()
    if not label:
        return []
    value = m.group("value").strip()
    return [
        AttributeCandidate(
            raw_name=label,
            chunk_id=chunk.chunk_id,
            block_id=block.block_id,
            source_text=chunk.source_text,
            origin="form_field",
            cells={"description": value} if value else {},
        )
    ]


def _prose_candidates(
    block: Block, chunk: Chunk, known_entities: Mapping[str, KnownEntity], lexicons: Lexicons
) -> list[AttributeCandidate]:
    if block.kind not in ("paragraph", "list_item") or not block.text:
        return []
    text = block.text
    out: list[AttributeCandidate] = []
    seen: set[str] = set()

    def add(raw_name: str, sentence: str) -> None:
        key = raw_name.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        out.append(
            AttributeCandidate(
                raw_name=raw_name.strip(),
                chunk_id=chunk.chunk_id,
                block_id=block.block_id,
                source_text=chunk.source_text,
                origin="prose_regex",
                sentence=sentence,
            )
        )

    for name in known_entities:
        if re.search(rf"\b{re.escape(name)}s?\b", text, re.IGNORECASE):
            add(name, merged_span_for(text, name) or text)

    for canonical, expansions in lexicons.gazetteer.items():
        for term in (canonical, *expansions):
            if re.search(rf"\b{re.escape(term)}s?\b", text, re.IGNORECASE):
                add(term, merged_span_for(text, term) or text)
                break

    patterns = compile_patterns(list(known_entities))
    for sentence in split_sentences(text):
        for m in patterns["rfc2119"].finditer(sentence):
            for item in split_rfc2119_tail(m.group("tail")):
                add(item, merged_span_for(text, item) or sentence)
        first_word = _SENTENCE_INITIAL_WORD_RE.match(sentence)
        if first_word and _NOMINALISATION_SUFFIX_RE.search(first_word.group(1)):
            word = first_word.group(1)
            add(word, merged_span_for(text, word) or sentence)

    return out


def generate_candidates(
    chunk: Chunk, document: Document, lexicons: Lexicons, cfg: AppConfig
) -> list[AttributeCandidate]:
    known_entities = known_entities_for_document(document, lexicons)
    block_map = document.block_map()
    out: list[AttributeCandidate] = []
    for block_id in chunk.block_ids:
        block = block_map.get(block_id)
        if block is None or block.kind == "furniture":
            continue
        if block.kind == "table":
            out.extend(_spec_table_candidates(block, chunk, cfg))
            out.extend(_kv_table_candidates(block, chunk, cfg))
        elif block.kind == "form_field":
            out.extend(_form_field_candidates(block, chunk))
        else:
            out.extend(_prose_candidates(block, chunk, known_entities, lexicons))
    return out


__all__ = ["generate_candidates"]
