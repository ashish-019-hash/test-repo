"""Token counting for chunk sizing.

Uses ``tiktoken``'s ``cl100k_base`` encoding when the optional dependency is
installed, and falls back to a deterministic word-count heuristic otherwise so
chunking behaves identically (modulo count precision) with or without the
dependency present.
"""

from __future__ import annotations

import math

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:  # pragma: no cover - exercised when tiktoken is absent
    _ENCODING = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return math.ceil(len(text.split()) * 1.3)


__all__ = ["count_tokens"]
