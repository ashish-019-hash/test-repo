"""Deterministic scoring engine: candidate generation, signal scoring, and entity binding."""

from doc_extractor.scoring.binding import bind_entity, heading_implied_entity, known_entities_for_document
from doc_extractor.scoring.candidates import generate_candidates
from doc_extractor.scoring.engine import score
from doc_extractor.scoring.lexicons import KnownEntity, Lexicons, load_lexicons
from doc_extractor.scoring.normalize import (
    aliases_for,
    attribute_type_for,
    camel_case,
    optionality_for,
    unit_for,
    value_domain_for,
)
from doc_extractor.scoring.signals import ScoringContext

__all__ = [
    "KnownEntity",
    "Lexicons",
    "ScoringContext",
    "aliases_for",
    "attribute_type_for",
    "bind_entity",
    "camel_case",
    "generate_candidates",
    "heading_implied_entity",
    "known_entities_for_document",
    "load_lexicons",
    "optionality_for",
    "score",
    "unit_for",
    "value_domain_for",
]
