"""De-hyphenation, whitespace collapsing, and furniture (repeated/regex-matched line) marking.

Furniture blocks are flagged, never deleted, so provenance stays intact.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import yaml

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.document import Block

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_WHITESPACE_RE = re.compile(r"[ \t]+")


def dehyphenate(text: str) -> str:
    """Join a word broken across a line by a trailing hyphen: 'infor-\\nmation' -> 'information'."""
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def load_furniture_patterns(cfg: AppConfig) -> list[str]:
    path = cfg.lexicon_paths["furniture_patterns"]
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("patterns", []))


def mark_furniture(blocks: list[Block], patterns: Iterable[str]) -> list[Block]:
    """Flag blocks that look like page furniture:
    - text matching one of `patterns` (case-insensitive regexes), or
    - text that repeats as the first or last non-empty block of >= 2 distinct pages.
    """
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)

    first_texts: dict[str, set[int]] = {}
    last_texts: dict[str, set[int]] = {}
    for page, page_blocks in by_page.items():
        non_empty = [b for b in page_blocks if b.text.strip()]
        if not non_empty:
            continue
        first_texts.setdefault(non_empty[0].text.strip(), set()).add(page)
        last_texts.setdefault(non_empty[-1].text.strip(), set()).add(page)
    repeated = {t for t, pages in first_texts.items() if len(pages) >= 2}
    repeated |= {t for t, pages in last_texts.items() if len(pages) >= 2}

    out: list[Block] = []
    for b in blocks:
        stripped = b.text.strip()
        is_furniture = (
            b.kind == "furniture" or stripped in repeated or any(p.search(stripped) for p in compiled)
        )
        if is_furniture and b.kind != "furniture":
            b = b.model_copy(update={"kind": "furniture"})
        out.append(b)
    return out
