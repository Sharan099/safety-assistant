import datetime

from safety_assistant.ingestion.normalize import normalize_generic, normalize_regulation, parse_cover
from safety_assistant.ingestion.parse.contract import ParsedPage

COVER = """E/ECE/324/Rev.1/Add.93/Rev.4
Addendum 93: Regulation No. 94
Revision 4
Incorporating all valid text up to:
Supplement 1 to the 03 series of amendments – Date of entry into force: 28 May 2019
Supplement 2 to the 03 series of amendments – Date of entry into force: 3 January 2021
04 series of amendments to the UN Regulation – Date of entry into force 09 June 2021--
E/ECE/324/Rev.1/Add.93/Rev.4−E/ECE/TRANS/505/Rev.1/Add.93/Rev.4
29 December 2022
"""


def test_cover_metadata_from_real_r94_layout() -> None:
    c = parse_cover([COVER])
    assert c.document_symbol == "E/ECE/324/Rev.1/Add.93/Rev.4"
    assert c.revision == "Rev.4"
    assert [a.entry_into_force for a in c.amendments] == [
        datetime.date(2019, 5, 28),
        datetime.date(2021, 1, 3),
        datetime.date(2021, 6, 9),
    ]
    assert c.latest_entry_into_force == datetime.date(2021, 6, 9) and c.latest_series == "04"
    assert c.document_date == datetime.date(2022, 12, 29)


def _page(n: int, text: str) -> ParsedPage:
    return ParsedPage(page_number=n, text=text, char_count=len(text), text_quality=1.0, needs_ocr=False)


BODY = """E/ECE/324/Rev.1/Add.93/Rev.4
5
1.
Scope
This Regulation applies to vehicles of category M1.
2.
Definitions
2.1.
"Protective system" means interior fittings and devices intended to restrain the occupants.
2.2.
"Type of protective system" means a category of protective devices.
3.
Application for approval
3.1.
The application shall be submitted by the vehicle manufacturer.
3.1.1.
It shall be accompanied by drawings. See Annex 3, paragraph 1.4.3. and paragraph 2.1. above.
3
2.6.
A footnote marker above must not open clause 3 again.
"""
ANNEX = """E/ECE/324/Rev.1/Add.93/Rev.4
Annex 3
23
Test procedure
1.
Installation
1.1.
The vehicle shall overlap the barrier face by 40 per cent ± 20 mm.
"""


def test_regulation_structure_paths_kinds_titles_and_xrefs() -> None:
    nd = normalize_regulation([_page(5, BODY), _page(23, ANNEX)])
    by = nd.by_path()
    assert by["1"].title == "Scope" and by["1"].depth == 1
    assert by["2.1"].kind == "DEFINITION" and by["2.1"].title == "Protective system"
    assert by["3.1.1"].parent_path == "3.1" and by["3.1.1"].normative is True
    assert by["annex-3/1.1"].annex == "Annex 3" and by["annex-3/1.1"].depth == 2
    targets = {(x.from_path, x.target_path) for x in nd.cross_references}
    assert ("3.1.1", "annex-3/1.4.3") in targets
    assert ("3.1.1", "2.1") in targets
    # the footnote marker "3" followed by "2.6." did not create a second clause 3
    assert [s.path for s in nd.sections if s.path == "3"] == ["3"]
    assert "2.6" not in by  # 2.6 is not a plausible successor of 3.1.1 → stays text


def test_parsed_hash_is_stable_and_content_sensitive() -> None:
    a = normalize_regulation([_page(5, BODY)]).parsed_hash
    b = normalize_regulation([_page(5, BODY)]).parsed_hash
    c = normalize_regulation([_page(5, BODY.replace("category M1", "category N1"))]).parsed_hash
    assert a == b != c


def test_toc_pages_stay_in_front_matter() -> None:
    toc = "Contents\n1. Scope ............ 5\n2. Definitions ....... 5\n3. Application ....... 9\n"
    nd = normalize_regulation([_page(3, toc), _page(5, BODY)])
    assert nd.sections[0].kind == "FRONT_MATTER" and "Scope ....." in nd.sections[0].content
    assert nd.by_path()["1"].title == "Scope"


def test_generic_normalizer_for_manuals() -> None:
    text = "1 INTRODUCTION\nSome intro text.\n1.1 Purpose\nWhy this manual exists.\nCHAPTER SUMMARY\nDone."
    nd = normalize_generic([_page(1, text)])
    paths = [s.path for s in nd.sections]
    assert "1" in paths and "1.1" in paths and nd.by_path()["1.1"].parent_path == "1"
