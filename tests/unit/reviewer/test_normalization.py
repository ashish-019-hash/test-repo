"""Tests for `doc_extractor.reviewer.normalization.normalize_entity_name`."""

from __future__ import annotations

from doc_extractor.reviewer.normalization import normalize_entity_name

ALIASES = {
    "EVC": ["Ethernet Virtual Connection", "Ethernet Virtual Circuit"],
    "UNI": ["User Network Interface", "User-Network Interface"],
}


def test_casefold_only() -> None:
    normalized, steps = normalize_entity_name("EVC", ALIASES)
    assert normalized == "evc"
    assert steps == ["casefold"]


def test_alias_of_bare_expansion() -> None:
    normalized, steps = normalize_entity_name("Ethernet Virtual Connection", ALIASES)
    assert normalized == "evc"
    assert "alias of EVC" in steps


def test_alias_with_parenthetical_does_not_expand_past_casefold() -> None:
    # The parenthetical form is not itself a configured alias string, so it just casefolds.
    normalized, steps = normalize_entity_name("Ethernet Virtual Connection (EVC)", ALIASES)
    assert normalized == "ethernet virtual connection (evc)"
    assert steps == ["casefold"]


def test_hyphenated_alias_normalizes_to_canonical() -> None:
    normalized, steps = normalize_entity_name("User-Network Interface", ALIASES)
    assert normalized == "uni"
    assert "normalize hyphen/dash/slash spacing" in steps
    assert "alias of UNI" in steps


def test_strip_surrounding_quotes() -> None:
    normalized, steps = normalize_entity_name('"EVC"', ALIASES)
    assert normalized == "evc"
    assert "strip surrounding quotes/brackets" in steps


def test_singularize_plural_table_term() -> None:
    normalized, steps = normalize_entity_name("Service Types", ALIASES)
    assert normalized == "service type"
    assert any("singularize plural token" in s for s in steps)


def test_singularize_ies_suffix() -> None:
    normalized, _steps = normalize_entity_name("Policies", ALIASES)
    assert normalized == "policy"


def test_no_strip_protected_token() -> None:
    normalized, steps = normalize_entity_name("CoS", ALIASES)
    assert normalized == "cos"
    assert steps == ["casefold"]


def test_possessive_apostrophe_is_not_mangled() -> None:
    """Regression: naive trailing-"s" stripping on "subscriber's" must not produce "subscriber'"."""
    normalized, _steps = normalize_entity_name("Subscriber's", ALIASES)
    assert normalized == "subscriber's"
    assert not normalized.endswith("'")


def test_acronym_is_never_singularized() -> None:
    normalized, _steps = normalize_entity_name("SIMS", ALIASES)
    # All-uppercase acronym tokens are protected from singularization.
    assert normalized == "sims"


def test_no_change_still_returns_empty_step_list_when_already_canonical() -> None:
    normalized, steps = normalize_entity_name("evc", ALIASES)
    assert normalized == "evc"
    assert steps == []


def test_whitespace_and_trim() -> None:
    normalized, steps = normalize_entity_name("  EVC   ID  ", ALIASES)
    assert normalized == "evc id"
    assert "trim whitespace" in steps
    assert "collapse whitespace" in steps
