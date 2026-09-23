"""Unit tests for `scoring.normalize`: camelCase, units, value domains, types, aliases."""

from __future__ import annotations

import pytest

from doc_extractor.config.models import AppConfig
from doc_extractor.scoring.lexicons import Lexicons, load_lexicons
from doc_extractor.scoring.normalize import (
    aliases_for,
    attribute_type_for,
    camel_case,
    optionality_for,
    unit_for,
    value_domain_for,
)


@pytest.fixture(scope="module")
def lex(cfg: AppConfig) -> Lexicons:
    return load_lexicons(cfg)


# ---- camel_case ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_name, expected",
    [
        ("CIR", "cir"),
        ("Committed Information Rate", "committedInformationRate"),
        ("EVC ID", "evcId"),
        ("MAC Address", "macAddress"),
        ("CoS Name", "cosName"),
        ("MSISDN", "msisdn"),
    ],
)
def test_camel_case(raw_name: str, expected: str) -> None:
    assert camel_case(raw_name) == expected


def test_camel_case_empty() -> None:
    assert camel_case("") == ""


# ---- unit_for -------------------------------------------------------------------------------


def test_unit_for_mbps() -> None:
    unit = unit_for("Mbps")
    assert unit is not None
    assert unit.raw == "Mbps"
    assert unit.base == "bit/s"
    assert unit.factor == pytest.approx(1e6)


def test_unit_for_unknown_raw_kept() -> None:
    unit = unit_for("furlongs")
    assert unit is not None
    assert unit.raw == "furlongs"
    assert unit.base is None
    assert unit.factor is None


def test_unit_for_none_or_empty() -> None:
    assert unit_for(None) is None
    assert unit_for("") is None
    assert unit_for("   ") is None


# ---- value_domain_for -----------------------------------------------------------------------


def test_value_domain_for_range() -> None:
    vd = value_domain_for({"range": "10–1000"})
    assert vd is not None
    assert vd.kind == "range"
    assert vd.from_ == 10
    assert vd.to == 1000


def test_value_domain_for_enum() -> None:
    vd = value_domain_for({"range": "Point-to-Point, Multipoint-to-Multipoint"})
    assert vd is not None
    assert vd.kind == "enum"
    assert vd.values == ["Point-to-Point", "Multipoint-to-Multipoint"]


def test_value_domain_for_none_when_no_range_cell() -> None:
    assert value_domain_for({}) is None


# ---- attribute_type_for ---------------------------------------------------------------------


def test_attribute_type_for_from_type_cell(lex: Lexicons) -> None:
    assert attribute_type_for({"type": "Integer"}, lex, "CIR") == "integer"
    assert attribute_type_for({"type": "String"}, lex, "EVC ID") == "string"
    assert attribute_type_for({"type": "Enum"}, lex, "Service Type") == "enum"


def test_attribute_type_for_falls_back_to_property_noun(lex: Lexicons) -> None:
    assert attribute_type_for({}, lex, "Bandwidth") == "measure"
    assert attribute_type_for({}, lex, "Reference Code") == "identifier"


def test_attribute_type_for_unknown(lex: Lexicons) -> None:
    assert attribute_type_for({}, lex, "Zorblatt") == "unknown"


# ---- optionality_for -------------------------------------------------------------------------


@pytest.mark.parametrize("cell, expected", [("M", "M"), ("o", "O"), ("C", "C"), ("", None), ("X", None)])
def test_optionality_for(cell: str, expected: str | None) -> None:
    assert optionality_for({"optionality": cell}) == expected


def test_optionality_for_missing_key() -> None:
    assert optionality_for({}) is None


# ---- aliases_for -----------------------------------------------------------------------------


def test_aliases_for_cir_matches_description(lex: Lexicons) -> None:
    aliases = aliases_for("CIR", "Committed Information Rate for the EVC", lex)
    assert aliases == ["Committed Information Rate"]


def test_aliases_for_no_gazetteer_match(lex: Lexicons) -> None:
    assert aliases_for("Zorblatt", "some description", lex) == []


def test_aliases_for_non_acronym_name_skipped(lex: Lexicons) -> None:
    # Lower/mixed-case multi-word names are not acronym-like, so no expansion is attempted
    # even if by coincidence they alias a gazetteer entry.
    assert aliases_for("Committed Information Rate", "irrelevant", lex) == []
