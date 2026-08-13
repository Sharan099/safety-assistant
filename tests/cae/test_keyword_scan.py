from pathlib import Path

from packages.cae.keyword_scan import scan_file, scan_keywords

SAMPLE_DECK = """\
*KEYWORD
$ comment line, not a keyword
*PART
seat_frame
1,1,1
*SECTION_SHELL
1,2,0.0,0.0
*MAT_024
1,7.85e-9,2.1e5,0.3
*NODE
1,0.0,0.0,0.0
*ELEMENT_SHELL
1,1,1,2,3,4
*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE
*INCLUDE
vehicle.k
*INCLUDE
dummy.k
*END
"""


def test_scan_counts_keywords_and_roots() -> None:
    result = scan_keywords(SAMPLE_DECK)
    assert result.keyword_counts["PART"] == 1
    assert result.keyword_counts["MAT_024"] == 1
    assert result.keyword_counts["CONTACT_AUTOMATIC_SURFACE_TO_SURFACE"] == 1
    assert result.root_counts["PART"] == 1
    assert result.root_counts["MAT"] == 1
    assert result.root_counts["CONTACT"] == 1
    assert result.root_counts["SECTION"] == 1
    assert result.root_counts["NODE"] == 1
    assert result.root_counts["ELEMENT"] == 1
    assert result.include_count == 2


def test_scan_ignores_comment_lines() -> None:
    result = scan_keywords("$ *PART this is a comment, not a keyword\n*NODE\n")
    assert "PART" not in result.keyword_counts
    assert result.root_counts["NODE"] == 1


def test_scan_file_reads_and_derives_model_hint(tmp_path: Path) -> None:
    deck_path = tmp_path / "Seat_Model_Explicit.k"
    deck_path.write_text(SAMPLE_DECK, encoding="utf-8")
    result = scan_file(deck_path)
    assert "Seat" in result.model_hints
    assert result.root_counts["PART"] == 1


def test_scan_file_never_crashes_on_non_utf8_bytes(tmp_path: Path) -> None:
    deck_path = tmp_path / "legacy_fixed_width.k"
    deck_path.write_bytes(b"*NODE\n1,0.0,\xff\xfe,0.0\n")
    result = scan_file(deck_path)
    assert result.root_counts.get("NODE") == 1


def test_unknown_keyword_not_bucketed_into_any_root() -> None:
    result = scan_keywords("*SOME_FUTURE_KEYWORD_NOBODY_KNOWS\n")
    assert result.keyword_counts["SOME_FUTURE_KEYWORD_NOBODY_KNOWS"] == 1
    assert result.root_counts == {}
