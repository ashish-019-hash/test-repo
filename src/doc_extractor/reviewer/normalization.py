"""Pure entity-name normalization: `normalize_entity_name`.

No merging, no dropping — every entity gets exactly one normalized record. Each
applied transformation is recorded as a human-readable step string so the
Entity Normalization Agent can build a `reasoning` sentence from it.
"""

from __future__ import annotations

import re
import unicodedata

_WRAP_PAIRS: dict[str, str] = {
    '"': '"',
    "'": "'",
    "(": ")",
    "[": "]",
    "{": "}",
    "\u201c": "\u201d",
    "\u2018": "\u2019",
}
_DASH_SLASH_RE = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014\u2015/]+")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?]+$")
_WHITESPACE_RE = re.compile(r"\s+")

# Conservative no-strip set: normalized (casefolded) tokens that end in "s" but are not
# plurals, or where singularising would destroy meaning (abbreviations, status words).
_NO_STRIP_TOKENS: frozenset[str] = frozenset(
    {"cos", "status", "address", "class", "series", "access", "business", "gas", "bus"}
)


def _strip_wrapping(s: str) -> str:
    while len(s) >= 2 and s[0] in _WRAP_PAIRS and s[-1] == _WRAP_PAIRS[s[0]]:
        s = s[1:-1].strip()
    return s


def _despace_dashes(s: str) -> str:
    despaced = _DASH_SLASH_RE.sub(" ", s)
    return _WHITESPACE_RE.sub(" ", despaced).strip()


def _pre_fold_pipeline(name: str) -> str:
    """Steps 1-6, without casefolding or singularisation: used to compare/derive canonical forms."""
    s = unicodedata.normalize("NFKC", name).strip()
    s = _WHITESPACE_RE.sub(" ", s)
    s = _strip_wrapping(s)
    s = _TRAILING_PUNCT_RE.sub("", s).strip()
    s = _despace_dashes(s)
    return s


def _singularize_token(token: str, is_acronym: bool) -> str:
    if is_acronym or len(token) <= 3 or token in _NO_STRIP_TOKENS or "'" in token:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _singularize(pre_fold: str, folded: str) -> tuple[str, bool]:
    pre_tokens = pre_fold.split(" ")
    fold_tokens = folded.split(" ")
    if len(pre_tokens) != len(fold_tokens):
        return folded, False
    out_tokens = []
    changed = False
    for pre_tok, fold_tok in zip(pre_tokens, fold_tokens, strict=True):
        is_acronym = pre_tok.isupper() and len(pre_tok) > 1
        singular = _singularize_token(fold_tok, is_acronym)
        if singular != fold_tok:
            changed = True
        out_tokens.append(singular)
    return " ".join(out_tokens), changed


def normalize_entity_name(name: str, aliases: dict[str, list[str]]) -> tuple[str, list[str]]:
    """Normalize `name`; returns (normalized, steps). Deterministic and non-destructive."""
    steps: list[str] = []
    s = name

    nfkc = unicodedata.normalize("NFKC", s)
    if nfkc != s:
        steps.append("NFKC unicode normalize")
    s = nfkc

    trimmed = s.strip()
    if trimmed != s:
        steps.append("trim whitespace")
    s = trimmed

    collapsed = _WHITESPACE_RE.sub(" ", s)
    if collapsed != s:
        steps.append("collapse whitespace")
    s = collapsed

    unwrapped = _strip_wrapping(s)
    if unwrapped != s:
        steps.append("strip surrounding quotes/brackets")
    s = unwrapped

    depunct = _TRAILING_PUNCT_RE.sub("", s).strip()
    if depunct != s:
        steps.append("strip trailing punctuation")
    s = depunct

    despaced = _despace_dashes(s)
    if despaced != s:
        steps.append("normalize hyphen/dash/slash spacing")
    pre_fold = despaced
    s = despaced

    folded = s.casefold()
    if folded != s:
        steps.append("casefold")
    s = folded

    singular, changed = _singularize(pre_fold, folded)
    if changed:
        steps.append(f"singularize plural token(s) ({folded} -> {singular})")
    s = singular

    canon, alias_step = _alias_canonicalize(s, aliases)
    if alias_step is not None:
        steps.append(alias_step)
        s = canon

    return s, steps


def _alias_canonicalize(s: str, aliases: dict[str, list[str]]) -> tuple[str, str | None]:
    """If `s` equals a configured alias of canonical K, canonicalize to K (casefolded)."""
    for canonical, alist in aliases.items():
        canon_norm = _pre_fold_pipeline(canonical).casefold()
        if s == canon_norm:
            continue  # already canonical form: no step needed
        for alias in alist:
            alias_norm = _pre_fold_pipeline(alias).casefold()
            if s == alias_norm:
                return canon_norm, f"alias of {canonical}"
    return s, None
