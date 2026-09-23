"""Entity Reviewer helper modules: normalization, duplicate similarity, quality scoring.

`load_lexicons` is re-exported here so agents can do:

    from doc_extractor.reviewer import load_lexicons
"""

from __future__ import annotations

from doc_extractor.scoring.lexicons import load_lexicons

__all__ = ["load_lexicons"]
