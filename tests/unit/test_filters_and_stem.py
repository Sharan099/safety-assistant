import datetime

from safety_assistant.retrieval.filters import (
    ScopeFilter,
    is_relevant,
    light_stem,
    shared_term_count,
    significant_tokens,
)


def test_light_stem_conservative() -> None:
    assert light_stem("doors") == "door" and light_stem("categories") == "category"
    assert light_stem("masses") == "mass" and light_stem("exceeded") == "exceed"
    assert light_stem("hic15") == "hic15" and light_stem("mass") == "mass" and light_stem("bus") == "bus"


def test_relevance_floor_uses_stems_and_short_query_rule() -> None:
    assert is_relevant("Are the doors allowed to open?", "During the test no door shall open.")
    assert is_relevant("ISOFIX definition", '"ISOFIX" is a system for the connection of child restraint systems')
    assert not is_relevant("tibia index limit", "The buckle shall remain closed whatever the position of the vehicle.")
    assert is_relevant("", "anything")  # nothing to check against passes through


def test_significant_tokens_drop_stopwords() -> None:
    assert significant_tokens("what is the limit of the tibia") == {"limit", "tibia"}
    assert shared_term_count("tibia index", "the tibia index shall") == 2


def test_scope_effective_date_defaults_to_today() -> None:
    assert ScopeFilter(as_of=datetime.date(2020, 1, 1)).effective_date() == datetime.date(2020, 1, 1)
    assert ScopeFilter().effective_date(datetime.date(2026, 9, 11)) == datetime.date(2026, 9, 11)


def test_regulation_scope_includes_amendment_sheets() -> None:
    from safety_assistant.retrieval.base import key_in_scope

    assert key_in_scope("UN-R94", ("UN-R94",))
    assert key_in_scope("UN-R94-AMEND-05", ("UN-R94",))  # the sheet that amends the selected text
    assert not key_in_scope("UN-R95", ("UN-R94",))
    assert not key_in_scope("UN-R941", ("UN-R94",))


def test_definition_intent_and_synonym_rewrite() -> None:
    from safety_assistant.generation.grounding import rewrite_query
    from safety_assistant.retrieval.service import _DEFINITION_INTENT

    assert _DEFINITION_INTENT.search("ECRS definition") and _DEFINITION_INTENT.search("What is a protective system?")
    assert not _DEFINITION_INTENT.search("ThCC limit frontal?")
    assert rewrite_query("belt webbing min width") == "belt WEBBING (strap) min width"


def test_defines_term_matches_the_quoted_defined_phrase() -> None:
    from safety_assistant.retrieval.filters import defines_term

    assert defines_term("What is i-Size?", '2.3.1. "i-Size" means a category of Enhanced Child Restraint System')
    assert defines_term("ISOFIX definition", '2.28. "ISOFIX" means a system for the connection')
    # the term only appears inside a longer defined phrase → not this thing's definition
    assert not defines_term("ISOFIX definition", '2.29. "ISOFIX anchorage system" means a system')
    assert not defines_term("tibia index limit", '"Tibia index" means')  # no definition intent words
