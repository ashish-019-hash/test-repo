"""Typed application configuration (loaded from YAML + environment overrides)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from doc_extractor.exceptions import ConfigError

POSITIVE_SIGNALS: tuple[str, ...] = (
    "structural.spec_table_row",
    "structural.key_value_pair",
    "structural.attribute_heading",
    "lexical.property_noun_head",
    "lexical.gazetteer_hit",
    "value_domain.typed_value",
    "value_domain.enum_range_default",
    "value_domain.cardinality_marker",
    "syntactic.binding_pattern",
)
NEGATIVE_SIGNALS: tuple[str, ...] = (
    "negative.has_sub_attributes",
    "negative.lifecycle_subject",
    "negative.counted_instances",
    "negative.verb_nominalisation",
    "negative.furniture",
)
ALL_SIGNALS: tuple[str, ...] = POSITIVE_SIGNALS + NEGATIVE_SIGNALS


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMConfig(_Cfg):
    provider: Literal["auto", "azure", "rules", "mock"] = "auto"
    temperature: float = 0.0
    seed: int = 42
    cache_dir: str | None = None
    max_output_tokens: int = 2000
    timeout_seconds: float = 60.0
    replay_dir: str | None = Field(
        default=None, description="Fixtures directory for the mock/replay provider."
    )
    replay_strict: bool = False


class IngestConfig(_Cfg):
    min_chars_per_page: int = 20
    ocr_page_ratio: float = 0.5
    ocr_enabled: bool = False


class ChunkingConfig(_Cfg):
    strategy: Literal["by_page", "fixed_tokens", "heading_aware"] = "heading_aware"
    max_tokens: int = 800
    overlap_tokens: int = 100
    heading_split_level: int = 3
    sentence_regex: str = r"(?<=[.!?])\s+(?=[A-Z])"


class ClampConfig(_Cfg):
    min: float = 0.0
    max: float = 1.0


class ShortCircuitConfig(_Cfg):
    enabled: bool = True
    confidence: float = 1.0
    min_populated_cells: int = 3
    max_name_tokens: int = 6


class RoutingConfig(_Cfg):
    accept: float = 0.85
    review: float = 0.50

    @model_validator(mode="after")
    def _order(self) -> RoutingConfig:
        if not self.review < self.accept:
            raise ConfigError("scoring.routing.review must be < scoring.routing.accept")
        return self


class ScoringConfig(_Cfg):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "structural.spec_table_row": 0.45,
            "structural.key_value_pair": 0.30,
            "structural.attribute_heading": 0.20,
            "lexical.property_noun_head": 0.20,
            "lexical.gazetteer_hit": 0.15,
            "value_domain.typed_value": 0.20,
            "value_domain.enum_range_default": 0.15,
            "value_domain.cardinality_marker": 0.15,
            "syntactic.binding_pattern": 0.20,
            "negative.has_sub_attributes": -0.40,
            "negative.lifecycle_subject": -0.30,
            "negative.counted_instances": -0.25,
            "negative.verb_nominalisation": -0.25,
            "negative.furniture": -1.0,
        }
    )
    clamp: ClampConfig = ClampConfig()
    short_circuit: ShortCircuitConfig = ShortCircuitConfig()
    routing: RoutingConfig = RoutingConfig()
    spec_table_headers: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "name": ["name", "attribute", "parameter", "field", "attribute name", "parameter name"],
            "type": ["type", "datatype", "data type"],
            "units": ["unit", "units"],
            "optionality": ["m/o/c", "moc", "optionality", "mandatory", "opt", "m/o"],
            "default": ["default", "default value"],
            "description": ["description", "desc", "definition"],
            "range": ["range", "values", "allowed values", "domain", "value range", "enumeration"],
        }
    )
    attribute_heading_regex: str = r"attribute|parameter|characteristic|field|property|data element"

    @model_validator(mode="after")
    def _weights(self) -> ScoringConfig:
        missing = [k for k in ALL_SIGNALS if k not in self.weights]
        if missing:
            raise ConfigError(f"scoring.weights missing keys: {missing}")
        unknown = [k for k in self.weights if k not in ALL_SIGNALS]
        if unknown:
            raise ConfigError(f"scoring.weights has unknown keys: {unknown}")
        for k in POSITIVE_SIGNALS:
            if self.weights[k] <= 0:
                raise ConfigError(f"scoring.weights.{k} must be > 0")
        for k in NEGATIVE_SIGNALS:
            if self.weights[k] >= 0:
                raise ConfigError(f"scoring.weights.{k} must be < 0")
        if not self.clamp.min < self.clamp.max:
            raise ConfigError("scoring.clamp.min must be < scoring.clamp.max")
        return self


class BindingConfig(_Cfg):
    scope_multipliers: dict[str, float] = Field(
        default_factory=lambda: {
            "table_subject": 1.0,
            "nearest_heading": 0.9,
            "syntactic": 0.85,
            "compound_modifier": 0.7,
            "document_default": 0.5,
        }
    )
    distance_decay: float = 0.05
    floor: float = 0.5
    document_default_entity: str | None = None


class EntityGenerationConfig(_Cfg):
    promotion_min_sub_attributes: int = 2
    min_mentions: int = 2
    min_confidence: float = 0.5
    origin_weights: dict[str, float] = Field(
        default_factory=lambda: {"binding": 0.9, "promotion": 0.85, "lexicon": 0.6, "llm": 0.7}
    )


class DuplicateConfig(_Cfg):
    duplicate_threshold: float = 0.90
    possible_threshold: float = 0.70
    require_type_match: bool = True
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "name_sim": 0.20,
            "normalized_exact": 0.25,
            "type_match": 0.15,
            "chunk_overlap": 0.15,
            "attribute_overlap": 0.10,
            "evidence_overlap": 0.05,
            "alias_link": 0.10,
        }
    )

    @model_validator(mode="after")
    def _order(self) -> DuplicateConfig:
        if not self.possible_threshold < self.duplicate_threshold:
            raise ConfigError("reviewer.duplicate.possible_threshold must be < duplicate_threshold")
        return self


class QualityConfig(_Cfg):
    accept: float = 0.75
    review: float = 0.50
    llm_adjustment_max: float = 0.2
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "specificity": 0.125,
            "uniqueness": 0.125,
            "real_world_correspondence": 0.125,
            "contextual_relevance": 0.125,
            "completeness": 0.125,
            "source_grounding": 0.125,
            "entity_type_correctness": 0.125,
            "disambiguation_potential": 0.125,
        }
    )

    @model_validator(mode="after")
    def _checks(self) -> QualityConfig:
        if not self.review < self.accept:
            raise ConfigError("reviewer.quality.review must be < reviewer.quality.accept")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"reviewer.quality.weights must sum to 1.0 (got {total})")
        return self


class ReviewerConfig(_Cfg):
    duplicate: DuplicateConfig = DuplicateConfig()
    quality: QualityConfig = QualityConfig()


class MappingConfig(_Cfg):
    min_confidence: float = 0.5
    include_review_entities: bool = False


class LexiconsConfig(_Cfg):
    property_nouns: str = "lexicons/property_nouns.yaml"
    gazetteer: str = "lexicons/telecom_gazetteer.yaml"
    sid_abes: str = "lexicons/sid_abes.yaml"
    known_entities: str = "lexicons/known_entities.yaml"
    aliases: str = "lexicons/aliases.yaml"
    furniture_patterns: str = "lexicons/furniture_patterns.yaml"
    lifecycle_verbs: str = "lexicons/lifecycle_verbs.yaml"
    generic_terms: list[str] = Field(
        default_factory=lambda: [
            "system",
            "data",
            "information",
            "value",
            "item",
            "thing",
            "object",
            "element",
        ]
    )

    def file_fields(self) -> dict[str, str]:
        return {
            "property_nouns": self.property_nouns,
            "gazetteer": self.gazetteer,
            "sid_abes": self.sid_abes,
            "known_entities": self.known_entities,
            "aliases": self.aliases,
            "furniture_patterns": self.furniture_patterns,
            "lifecycle_verbs": self.lifecycle_verbs,
        }


class RetryConfig(_Cfg):
    max_attempts: int = 3
    backoff_seconds: float = 0.5


class OutputConfig(_Cfg):
    float_precision: int = 4
    graph_export: Literal["json", "graphml"] | None = None


class LoggingConfig(_Cfg):
    level: str = "INFO"
    format: Literal["console", "json"] = "console"


class AppConfig(_Cfg):
    llm: LLMConfig = LLMConfig()
    ingest: IngestConfig = IngestConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    scoring: ScoringConfig = ScoringConfig()
    binding: BindingConfig = BindingConfig()
    entity_generation: EntityGenerationConfig = EntityGenerationConfig()
    reviewer: ReviewerConfig = ReviewerConfig()
    mapping: MappingConfig = MappingConfig()
    lexicons: LexiconsConfig = LexiconsConfig()
    retry: RetryConfig = RetryConfig()
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()

    # Resolved at load time (not part of YAML); excluded from the config hash payload.
    lexicon_paths: dict[str, Path] = Field(default_factory=dict, exclude=True)
    config_hash: str = Field(default="", exclude=True)
