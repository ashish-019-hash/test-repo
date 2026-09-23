"""Loads the YAML lexicons into typed, lookup-friendly structures.

Everything here is pure/derived from `cfg.lexicon_paths`; no state is shared across
documents except what is passed back in the `Lexicons` object.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from doc_extractor.config.models import AppConfig
from doc_extractor.exceptions import ConfigError


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass(frozen=True)
class KnownEntity:
    name: str
    layer: str | None
    type: str | None
    aliases: tuple[str, ...] = ()


@dataclass
class Lexicons:
    known_entities: dict[str, KnownEntity]
    aliases: dict[str, list[str]]
    generic_terms: set[str]
    gazetteer: dict[str, list[str]]
    sid_abes: dict[str, dict[str, str]]
    property_nouns: dict[str, list[str]]
    furniture_patterns: list[re.Pattern[str]]
    lifecycle_verbs: set[str]
    attribute_heading_regex: re.Pattern[str]

    _entity_lookup: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    _property_noun_lookup: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    _gazetteer_lookup: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    _alias_lookup: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        entity_lookup: dict[str, str] = {}
        for name, ke in self.known_entities.items():
            entity_lookup[_norm(name)] = name
            for a in ke.aliases:
                entity_lookup[_norm(a)] = name
        for name, aliases in self.aliases.items():
            entity_lookup.setdefault(_norm(name), name)
            for a in aliases:
                entity_lookup.setdefault(_norm(a), name)
        self._entity_lookup = entity_lookup

        prop_lookup: dict[str, str] = {}
        for kind, nouns in self.property_nouns.items():
            for n in nouns:
                prop_lookup[_norm(n)] = kind
        self._property_noun_lookup = prop_lookup

        gaz_lookup: dict[str, str] = {}
        for name, aliases in self.gazetteer.items():
            gaz_lookup[_norm(name)] = name
            for a in aliases:
                gaz_lookup[_norm(a)] = name
        self._gazetteer_lookup = gaz_lookup

        alias_lookup: dict[str, str] = {}
        for name, aliases in self.aliases.items():
            alias_lookup[_norm(name)] = name
            for a in aliases:
                alias_lookup[_norm(a)] = name
        self._alias_lookup = alias_lookup

    # ---- lookups -----------------------------------------------------------------------
    def known_entity_lookup(self, name: str) -> KnownEntity | None:
        """Case-insensitive, alias-aware lookup of a known entity by name or alias."""
        canonical = self._entity_lookup.get(_norm(name))
        if canonical is None:
            return None
        return self.known_entities.get(canonical)

    def is_known_entity_name(self, name: str) -> bool:
        return _norm(name) in self._entity_lookup

    def property_noun_kind(self, word: str) -> str | None:
        return self._property_noun_lookup.get(_norm(word))

    def is_property_noun(self, word: str) -> bool:
        return _norm(word) in self._property_noun_lookup

    def gazetteer_match(self, text: str) -> str | None:
        """Return the canonical gazetteer key if `text` equals a gazetteer name/alias."""
        return self._gazetteer_lookup.get(_norm(text))

    def alias_canonical(self, name: str) -> str | None:
        return self._alias_lookup.get(_norm(name))

    def is_furniture(self, text: str) -> bool:
        return any(p.search(text) for p in self.furniture_patterns)

    def is_lifecycle_verb(self, word: str) -> bool:
        return _norm(word) in self.lifecycle_verbs


def load_lexicons(cfg: AppConfig) -> Lexicons:
    paths: Mapping[str, Path] = cfg.lexicon_paths
    if not paths:
        raise ConfigError("cfg.lexicon_paths is empty; call load_config() first")

    known_raw = _load_yaml(paths["known_entities"])
    known_entities = {
        name: KnownEntity(
            name=name,
            layer=v.get("layer"),
            type=v.get("type"),
            aliases=tuple(v.get("aliases", []) or []),
        )
        for name, v in known_raw.items()
    }
    aliases_raw = _load_yaml(paths["aliases"])
    aliases = {name: list(v or []) for name, v in aliases_raw.items()}
    gazetteer_raw = _load_yaml(paths["gazetteer"])
    gazetteer = {name: list(v or []) for name, v in gazetteer_raw.items()}
    sid_abes = _load_yaml(paths["sid_abes"])
    property_nouns_raw = _load_yaml(paths["property_nouns"])
    property_nouns = {kind: list(v or []) for kind, v in property_nouns_raw.items()}
    furniture_raw = _load_yaml(paths["furniture_patterns"])
    furniture_patterns = [re.compile(p, re.IGNORECASE) for p in furniture_raw.get("patterns", [])]
    lifecycle_raw = _load_yaml(paths["lifecycle_verbs"])
    lifecycle_verbs = {_norm(v) for v in lifecycle_raw.get("verbs", [])}
    attribute_heading_regex = re.compile(cfg.scoring.attribute_heading_regex, re.IGNORECASE)

    return Lexicons(
        known_entities=known_entities,
        aliases=aliases,
        generic_terms={_norm(t) for t in cfg.lexicons.generic_terms},
        gazetteer=gazetteer,
        sid_abes=sid_abes,
        property_nouns=property_nouns,
        furniture_patterns=furniture_patterns,
        lifecycle_verbs=lifecycle_verbs,
        attribute_heading_regex=attribute_heading_regex,
    )


__all__ = ["KnownEntity", "Lexicons", "load_lexicons"]
