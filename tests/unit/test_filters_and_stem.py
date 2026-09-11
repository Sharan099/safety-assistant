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
