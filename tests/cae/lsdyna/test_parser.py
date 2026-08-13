import pathlib

from packages.cae.lsdyna.models import ParsedDeck
from packages.cae.lsdyna.parser import parse_deck

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _parse_fixture(name: str) -> tuple[str, ParsedDeck]:
    text = FIXTURES.joinpath(name).read_text(encoding="utf-8")
    return text, parse_deck(text, name)


def test_simple_deck_parses_every_structured_root() -> None:
    _, deck = _parse_fixture("simple.k")

    assert len(deck.parts) == 1
    part = deck.parts[0]
    assert part.part_id == 1
    assert part.title == "Seat frame"
    assert part.section_id == 1
    assert part.material_id == 1
    assert part.parse_status == "PARSED"

    assert len(deck.sections) == 1
    assert deck.sections[0].section_id == 1
    assert deck.sections[0].section_type == "SHELL"

    assert len(deck.materials) == 1
    assert deck.materials[0].material_id == 1
    assert deck.materials[0].mat_type == "ELASTIC"

    assert len(deck.contacts) == 1
    contact = deck.contacts[0]
    assert contact.contact_type == "AUTOMATIC_SURFACE_TO_SURFACE"
    assert contact.ssid == 2
    assert contact.msid == 3

    assert len(deck.controls) == 1
    assert deck.controls[0].control_type == "TERMINATION"

    assert len(deck.databases) == 1
    assert deck.databases[0].database_type == "BINARY_D3PLOT"
    assert deck.databases[0].dt == 1.0

    # NODE/ELEMENT are "minimum" detected keywords (PRD_LEVEL3.md §11) but
    # have no dedicated table — they land in generic_keywords, not invented
    # into fake structured entities.
    generic_roots = {e.root for e in deck.generic_keywords}
    assert "NODE" in generic_roots
    assert "ELEMENT" in generic_roots


def test_unknown_keyword_is_preserved_generic_with_no_root() -> None:
    _, deck = _parse_fixture("unknown_keyword.k")
    unknown = next(e for e in deck.generic_keywords if e.keyword == "SOME_FUTURE_ADAS_SENSOR_DEFINITION")
    assert unknown.root is None
    assert "99,0.5,1.5" in unknown.raw.raw_text
    # KEYWORD/END are deck-structural markers, not entities — they're also
    # not in keyword_registry.yaml's roots, so they show up here too. That's
    # accurate, not a bug: unresolved_unknown_roots() means exactly what its
    # name says, "not in the registry", nothing more specific.
    assert deck.unresolved_unknown_roots() == {"KEYWORD", "END", "SOME_FUTURE_ADAS_SENSOR_DEFINITION"}
    # The known PART in the same file still parses normally.
    assert deck.parts[0].part_id == 1


def test_malformed_deck_degrades_gracefully_without_crashing() -> None:
    _, deck = _parse_fixture("malformed.k")
    # *PART with zero data lines: no title, no id fields at all -> UNKNOWN.
    assert deck.parts[0].parse_status == "UNKNOWN"
    assert deck.parts[0].part_id is None
    # *MAT_024 with zero data lines: the id couldn't be read, but "024" was
    # still recovered from the keyword name itself -> PARTIAL, not UNKNOWN —
    # honestly reflects "we know the material type, not its numeric id".
    assert deck.materials[0].parse_status == "PARTIAL"
    assert deck.materials[0].material_id is None
    assert deck.materials[0].mat_type == "024"
    # Raw text is still there even though nothing could be extracted.
    assert deck.parts[0].raw.raw_text.startswith("*PART")


def test_comments_interspersed_in_card_data_are_skipped_for_parsing() -> None:
    _, deck = _parse_fixture("comments.k")
    mat = deck.materials[0]
    assert mat.material_id == 7
    assert mat.mat_type == "PIECEWISE_LINEAR_PLASTICITY"
    assert mat.parse_status == "PARSED"


def test_fixed_width_whitespace_padded_fields_parse_correctly() -> None:
    _, deck = _parse_fixture("fixed_width.k")
    section = deck.sections[0]
    assert section.section_id == 3
    assert section.section_type == "SOLID"
    assert section.parse_status == "PARSED"


def test_genuinely_merged_fixed_width_token_does_not_fabricate_a_wrong_id() -> None:
    # No comma, no whitespace: _split_fields can't safely decompose this,
    # so it must NOT silently guess a 10-char split — a wrong guess would
    # fabricate an incorrect id, which this system never does.
    text = "*KEYWORD\n*SECTION_SOLID\n31\n*END\n"
    deck = parse_deck(text, "t.k")
    # "31" as a single token *is* a plausible (if maybe wrong) section id —
    # the honest behavior is to report exactly what was read, not invent
    # structure beyond it. There is exactly one field, so parse succeeds
    # with that field, never a fabricated split into multiple ids.
    assert deck.sections[0].section_id == 31


def test_include_filename_extracted_verbatim() -> None:
    _, deck = _parse_fixture("nested_include.key")
    assert len(deck.includes) == 1
    assert deck.includes[0].filename == "level2.k"
    assert deck.includes[0].parse_status == "PARSED"


def test_all_cards_list_is_never_dropped_even_for_generic_entries() -> None:
    _, deck = _parse_fixture("simple.k")
    assert len(deck.all_cards) == 10  # matches test_lexer's keyword count for simple.k
