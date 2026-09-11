import pathlib

from packages.cae.lsdyna.lexer import tokenize

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_tokenize_splits_into_one_card_per_keyword() -> None:
    text = FIXTURES.joinpath("simple.k").read_text(encoding="utf-8")
    cards = tokenize(text, "simple.k")
    keywords = [c.keyword for c in cards]
    assert keywords == [
        "KEYWORD",
        "PART",
        "SECTION_SHELL",
        "MAT_ELASTIC",
        "CONTACT_AUTOMATIC_SURFACE_TO_SURFACE",
        "CONTROL_TERMINATION",
        "DATABASE_BINARY_D3PLOT",
        "NODE",
        "ELEMENT_SHELL",
        "END",
    ]


def test_card_line_spans_are_1_indexed_and_contiguous() -> None:
    text = "*KEYWORD\n*PART\ntitle\n1,1,1\n*END\n"
    cards = tokenize(text, "t.k")
    by_kw = {c.keyword: c for c in cards}
    assert by_kw["KEYWORD"].line_start == 1
    assert by_kw["KEYWORD"].line_end == 1
    assert by_kw["PART"].line_start == 2
    assert by_kw["PART"].line_end == 4
    assert by_kw["END"].line_start == 5
    assert by_kw["END"].line_end == 5


def test_raw_text_preserves_everything_including_comments() -> None:
    text = FIXTURES.joinpath("comments.k").read_text(encoding="utf-8")
    cards = tokenize(text, "comments.k")
    mat_card = next(c for c in cards if c.keyword == "MAT_PIECEWISE_LINEAR_PLASTICITY")
    assert "$ material id" in mat_card.raw_text
    assert "$ trailing comment" in mat_card.raw_text
    # But data_lines() drops full-comment lines when looking for real data.
    assert all(not line.strip().startswith("$") for line in mat_card.data_lines())


def test_raw_hash_is_deterministic_and_content_addressed() -> None:
    cards_a = tokenize("*PART\ntitle\n1,1,1\n", "a.k")
    cards_b = tokenize("*PART\ntitle\n1,1,1\n", "b.k")  # same content, different file
    assert cards_a[0].raw_hash == cards_b[0].raw_hash

    cards_c = tokenize("*PART\ntitle\n2,2,2\n", "a.k")
    assert cards_a[0].raw_hash != cards_c[0].raw_hash


def test_pre_keyword_banner_content_is_dropped_not_attached() -> None:
    text = "this is a stray banner line before any keyword\n*END\n"
    cards = tokenize(text, "t.k")
    assert len(cards) == 1
    assert cards[0].keyword == "END"
    assert "banner" not in cards[0].raw_text


def test_empty_text_yields_no_cards() -> None:
    assert tokenize("", "empty.k") == []
