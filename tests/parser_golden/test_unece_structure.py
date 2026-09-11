"""Parser golden tests on real UNECE page text (fixtures/*.json, extracted from
the public consolidated texts). They pin the canonical structure the
normalizer must produce: clause paths, titles, definitions, annex scope,
cross-references, numeric/unit preservation and cover-page metadata."""

from __future__ import annotations

import datetime
import json
import pathlib

import pytest

from safety_assistant.ingestion.chunk import CitationContext, chunk_document
from safety_assistant.ingestion.normalize import normalize_regulation, parse_cover
from safety_assistant.ingestion.parse.contract import ParsedPage

FIX = pathlib.Path(__file__).parent / "fixtures"


def _pages(name: str) -> list[ParsedPage]:
    data = json.loads((FIX / name).read_text(encoding="utf-8"))["pages"]
    return [
        ParsedPage(page_number=int(n), text=t, char_count=len(t), text_quality=1.0, needs_ocr=False)
        for n, t in sorted(data.items(), key=lambda kv: int(kv[0]))
    ]


@pytest.fixture(scope="module")
def r94():  # type: ignore[no-untyped-def]
    return normalize_regulation(_pages("un_r94_rev4_pages.json"))


def test_r94_cover_page(r94) -> None:  # type: ignore[no-untyped-def]
    pages = _pages("un_r94_rev4_pages.json")
    cover = parse_cover([p.text for p in pages])
    assert cover.document_symbol == "E/ECE/324/Rev.1/Add.93/Rev.4"
    assert cover.revision == "Rev.4"
    assert cover.latest_entry_into_force == datetime.date(2021, 6, 9)
    assert cover.latest_series == "04"
    assert cover.document_date == datetime.date(2022, 12, 29)
    assert len(cover.amendments) == 3


def test_r94_body_clauses_titles_and_hierarchy(r94) -> None:  # type: ignore[no-untyped-def]
    by = r94.by_path()
    assert by["1"].title == "Scope"
    assert by["2"].title == "Definitions"
    assert by["3"].title == "Application for approval"
    assert by["3.2.1"].parent_path == "3.2" and by["3.2"].parent_path == "3"
    assert by["3.4.2"].depth == 3
    assert "paragraph 3.4.1. above" in by["3.4.2"].content


def test_r94_definitions_are_typed_and_titled(r94) -> None:  # type: ignore[no-untyped-def]
    by = r94.by_path()
    assert by["2.1"].kind == "DEFINITION" and by["2.1"].title == "Protective system"
    assert by["2.2"].kind == "DEFINITION" and by["2.2"].title == "Type of protective system"
    assert by["2.1"].normative is None


def test_r94_requirements_keep_numbers_units_and_operators(r94) -> None:  # type: ignore[no-untyped-def]
    by = r94.by_path()
    assert "exceed 1,3 at either location" in by["5.2.1.8"].content  # decimal comma preserved
    assert "shall not exceed 15 mm" in by["5.2.1.9"].content
    assert "shall not exceed 80 mm" in by["5.2.2"].content and "100 mm in the rearward" in by["5.2.2"].content
    assert by["5.2.1.8"].normative is True and by["5.2.3"].normative is True
    assert by["5.2.3"].content.startswith("During the test no door shall open.")


def test_r94_cross_references_resolve_into_annexes(r94) -> None:  # type: ignore[no-untyped-def]
    refs = {(x.from_path, x.target_path) for x in r94.cross_references}
    assert ("5.2.3.1.1", "annex-3/1.4.3.5.2.1") in refs
    assert ("5.2.3.1.2", "annex-3/1.4.3.5.2.2") in refs
    assert ("3.4.2", "3.4.1") in refs


def test_r94_annex_scope_restarts_numbering(r94) -> None:  # type: ignore[no-untyped-def]
    by = r94.by_path()
    assert by["annex-3"].kind == "ANNEX"
    assert by["annex-3/1.3.1"].annex == "Annex 3"
    assert "40 per cent" in by["annex-3/1.3.1"].content
    assert by["annex-9/2.1"].annex == "Annex 9" and by["annex-9/2.1"].depth == 2
    # body clause "1" and annex clause "1" are distinct nodes
    assert by["1"].path != by["annex-3/1"].path


def test_r94_toc_page_does_not_hijack_numbering(r94) -> None:  # type: ignore[no-untyped-def]
    front = r94.sections[0]
    assert front.kind == "FRONT_MATTER" and "Definition of deformable barrier" in front.content
    assert r94.by_path()["1"].page_start == 5


def test_r94_citation_labels_are_exact(r94) -> None:  # type: ignore[no-untyped-def]
    chunks = chunk_document(r94, [], CitationContext("UN-R94", "Rev.4 (04 series)"))
    labels = {c.citation_label for c in chunks}
    assert "UN R94 Rev.4 §5.2.2–5.2.3 (p. 13)" in labels  # tiny siblings merged, range is exact
    tib = next(c for c in chunks if c.citation_label.startswith("UN R94 Rev.4 §5.2.1.3–5.2.1.8"))
    assert tib.citation_label.startswith("UN R94 Rev.4 §5.2.1.")
    assert "5.2.1.8." in tib.content and "1,3" in tib.content


def test_r16_footnote_marker_and_wrapped_titles() -> None:
    nd = normalize_regulation(_pages("un_r16_rev7_pages.json"))
    by = nd.by_path()
    assert by["2.14.4"].title == "Emergency locking retractor (type 4)" and by["2.14.4"].kind == "DEFINITION"
    assert by["2.14.6"].title == "Belt adjustment device for height"
    assert by["3"].title == "Application for approval"  # the footnote "3." did not open a clause early
    assert by["3.1.2.1"].parent_path == "3.1.2"
    assert by["2.45"].content.startswith('"Vehicle is in normal operation" means')
