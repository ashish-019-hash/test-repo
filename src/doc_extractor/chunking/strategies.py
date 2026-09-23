"""Pure chunking strategies: ``list[Block] -> list[Chunk]``.

None of these functions read or write files; they only slice/join the blocks
they are given, per the ChunkingAgent contract ("must not change text, only
slice or join blocks").
"""

from __future__ import annotations

from doc_extractor.chunking.tokens import count_tokens
from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.document import Block
from doc_extractor.storage import ids


class _ChunkBuilder:
    """Assigns chunk ids from a running counter keyed by the chunk's first page."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self._counters: dict[int, int] = {}

    def build(self, blocks: list[Block], heading_path: list[str], strategy: str) -> Chunk:
        first_page = blocks[0].page
        last_page = blocks[-1].page
        n = self._counters.get(first_page, 0)
        self._counters[first_page] = n + 1
        source_text = "\n".join(b.text for b in blocks)
        return Chunk(
            chunk_id=ids.chunk_id(self.document_id, first_page, n),
            document_id=self.document_id,
            page_number=first_page,
            page_end=last_page,
            source_text=source_text,
            block_ids=[b.block_id for b in blocks],
            heading_path=list(heading_path),
            token_count=count_tokens(source_text),
            strategy=strategy,
        )


def by_page(blocks: list[Block], document_id: str, cfg: AppConfig) -> list[Chunk]:
    """One chunk per page.

    Tables are never split. If a page's combined text exceeds ``max_tokens``,
    tables are isolated into their own chunk (their rendered text already
    starts with the header row, so it is naturally "repeated" whenever the
    table has to stand alone) and the remaining text is packed into
    consecutive runs that each fit under the limit.
    """
    builder = _ChunkBuilder(document_id)
    max_tokens = cfg.chunking.max_tokens
    chunks: list[Chunk] = []

    pages = sorted({b.page for b in blocks})
    for page in pages:
        page_blocks = [b for b in blocks if b.page == page]
        if not page_blocks:
            continue
        total = count_tokens("\n".join(b.text for b in page_blocks))
        if total <= max_tokens:
            chunks.append(builder.build(page_blocks, [], "by_page"))
            continue

        run: list[Block] = []
        run_tokens = 0
        for b in page_blocks:
            if b.kind == "table":
                if run:
                    chunks.append(builder.build(run, [], "by_page"))
                    run, run_tokens = [], 0
                chunks.append(builder.build([b], [], "by_page"))
                continue
            t = count_tokens(b.text)
            if run and run_tokens + t > max_tokens:
                chunks.append(builder.build(run, [], "by_page"))
                run, run_tokens = [], 0
            run.append(b)
            run_tokens += t
        if run:
            chunks.append(builder.build(run, [], "by_page"))

    return chunks


def fixed_tokens(
    blocks: list[Block],
    document_id: str,
    cfg: AppConfig,
    *,
    heading_path: list[str] | None = None,
    strategy: str = "fixed_tokens",
    builder: _ChunkBuilder | None = None,
) -> list[Chunk]:
    """Pack consecutive blocks up to ``max_tokens``, carrying ``overlap_tokens`` forward.

    Blocks are the packing unit (never split inside a block, so never inside a
    sentence or a table row). Table blocks are always isolated into their own
    chunk. Overlap only carries non-table blocks, so a table is never
    duplicated into a neighbouring chunk.
    """
    builder = builder or _ChunkBuilder(document_id)
    heading_path = heading_path or []
    max_tokens = cfg.chunking.max_tokens
    overlap_tokens = cfg.chunking.overlap_tokens
    chunks: list[Chunk] = []

    current: list[Block] = []
    current_tokens = 0
    has_new = False

    def flush() -> None:
        nonlocal current, current_tokens, has_new
        if not current or not has_new:
            current, current_tokens, has_new = [], 0, False
            return
        chunks.append(builder.build(current, heading_path, strategy))
        carry: list[Block] = []
        carry_tokens = 0
        for b in reversed(current):
            if b.kind == "table":
                break
            t = count_tokens(b.text)
            if carry_tokens + t > overlap_tokens:
                break
            carry.insert(0, b)
            carry_tokens += t
        current, current_tokens, has_new = carry, carry_tokens, False

    for b in blocks:
        if b.kind == "table":
            flush()
            chunks.append(builder.build([b], heading_path, strategy))
            continue
        t = count_tokens(b.text)
        if current and current_tokens + t > max_tokens:
            flush()
        current.append(b)
        current_tokens += t
        has_new = True
    flush()

    return chunks


def heading_aware(blocks: list[Block], document_id: str, cfg: AppConfig) -> list[Chunk]:
    """Start a new chunk at every heading whose level is <= ``heading_split_level``.

    Within a section, blocks are packed with :func:`fixed_tokens`.
    ``heading_path`` is the breadcrumb stack of ancestor headings, including
    the heading that opened the current section. Furniture blocks (including
    headings reclassified as furniture, e.g. boilerplate section titles) are
    kept as ordinary content and never affect the heading stack.
    """
    builder = _ChunkBuilder(document_id)
    split_level = cfg.chunking.heading_split_level
    stack: dict[int, str] = {}
    section: list[Block] = []
    chunks: list[Chunk] = []

    def heading_path() -> list[str]:
        return [stack[level] for level in sorted(stack)]

    def flush_section() -> None:
        if not section:
            return
        chunks.extend(
            fixed_tokens(
                list(section),
                document_id,
                cfg,
                heading_path=heading_path(),
                strategy="heading_aware",
                builder=builder,
            )
        )
        section.clear()

    for b in blocks:
        if b.kind == "heading" and b.heading_level is not None:
            level = b.heading_level
            if level <= split_level:
                flush_section()
            for lvl in [k for k in stack if k >= level]:
                del stack[lvl]
            stack[level] = b.text
            section.append(b)
            continue
        section.append(b)

    flush_section()
    return chunks


__all__ = ["by_page", "fixed_tokens", "heading_aware"]
