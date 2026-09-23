"""camelCase naming, unit/value-domain parsing, attribute-type inference, alias extraction."""

from __future__ import annotations

import re
from typing import Any

from doc_extractor.schemas.attribute import AttributeType, Optionality
from doc_extractor.schemas.common import Unit, ValueDomain
from doc_extractor.scoring.lexicons import Lexicons
from doc_extractor.scoring.signals import head_noun

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

_UNIT_TABLE: dict[str, tuple[str, float]] = {
    "mbps": ("bit/s", 1e6),
    "gbps": ("bit/s", 1e9),
    "kbps": ("bit/s", 1e3),
    "bps": ("bit/s", 1.0),
    "ms": ("s", 1e-3),
    "s": ("s", 1.0),
    "bytes": ("byte", 1.0),
    "byte": ("byte", 1.0),
    "kb": ("byte", 1e3),
    "mb": ("byte", 1e6),
    "gb": ("byte", 1e9),
    "%": ("percent", 1.0),
}

_TYPE_CELL_MAP: dict[str, AttributeType] = {
    "integer": "integer",
    "int": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "str": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "enum": "enum",
    "date": "date",
    "identifier": "identifier",
}

_RANGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[–—-]\s*(\d+(?:\.\d+)?)\s*$")


def camel_case(name: str) -> str:
    """Acronym-aware camelCase: 'EVC ID' -> 'evcId', 'CIR' -> 'cir'."""
    tokens = _TOKEN_RE.findall(name)
    if not tokens:
        return ""
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            parts.append(tok.lower())
        elif tok.isupper() and len(tok) > 1:
            parts.append(tok[0].upper() + tok[1:].lower())
        else:
            parts.append(tok[0].upper() + tok[1:])
    return "".join(parts)


def unit_for(raw: str | None) -> Unit | None:
    if not raw or not raw.strip():
        return None
    raw_stripped = raw.strip()
    entry = _UNIT_TABLE.get(raw_stripped.lower())
    if entry is None:
        return Unit(raw=raw_stripped, base=None, factor=None)
    base, factor = entry
    return Unit(raw=raw_stripped, base=base, factor=factor)


def _numeric(text: str) -> int | float:
    return float(text) if "." in text else int(text)


def value_domain_for(cells: dict[str, str], text: str | None = None) -> ValueDomain | None:
    range_cell = cells.get("range", "").strip()
    if range_cell:
        if "," in range_cell:
            values = [v.strip() for v in range_cell.split(",") if v.strip()]
            return ValueDomain(kind="enum", values=values)
        m = _RANGE_RE.match(range_cell)
        if m:
            range_kwargs: dict[str, Any] = {
                "kind": "range",
                "from": _numeric(m.group(1)),
                "to": _numeric(m.group(2)),
            }
            return ValueDomain(**range_kwargs)
        return ValueDomain(kind="free", values=[range_cell])
    return None


def attribute_type_for(cells: dict[str, str], lexicons: Lexicons, raw_name: str) -> AttributeType:
    type_cell = cells.get("type", "").strip().lower()
    if type_cell in _TYPE_CELL_MAP:
        return _TYPE_CELL_MAP[type_cell]
    head = head_noun(raw_name)
    kind = lexicons.property_noun_kind(head)
    if kind == "measure":
        return "measure"
    if kind == "identifier":
        return "identifier"
    return "unknown"


def optionality_for(cells: dict[str, str]) -> Optionality | None:
    value = cells.get("optionality", "").strip().upper()
    if value in ("M", "O", "C"):
        return value  # type: ignore[return-value]
    return None


def aliases_for(raw_name: str, description: str | None, lexicons: Lexicons) -> list[str]:
    """Expand a short acronym-like name into its gazetteer expansion(s), preferring ones
    actually quoted in the candidate's description text."""
    compact = raw_name.replace(" ", "")
    if not (compact.isupper() and 2 <= len(compact) <= 8):
        return []
    canonical = lexicons.gazetteer_match(raw_name)
    if canonical is None:
        return []
    expansions = lexicons.gazetteer.get(canonical, [])
    if description:
        matched = [e for e in expansions if e.lower() in description.lower()]
        if matched:
            return matched
    return list(expansions)


__all__ = [
    "aliases_for",
    "attribute_type_for",
    "camel_case",
    "optionality_for",
    "unit_for",
    "value_domain_for",
]
