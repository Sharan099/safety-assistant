"""Tests for OCR-aware structure block extraction."""

from __future__ import annotations

from parser.structure_extract import pages_to_structured_blocks


def test_promotes_762_clause_out_of_annex_block():
    """OCR base PDFs embed §7.6.2 test text inside annex appendices — split on clause line."""
    pages = [
        {
            "page_number": 27,
            "text": (
                "Annex 3\n"
                "Diagram of apparatus for the tests specified in paragraph 7.6.1.1. above.\n"
                "7.6.2.\n"
                "Locking of emergency locking retractors\n"
                "7.6.2.1.\n"
                "The retractor shall be tested once for locking when the strap has been unwound.\n"
            ),
            "tables": [],
        }
    ]
    blocks = pages_to_structured_blocks(pages)
    section_ids = [b["section_id"] for b in blocks]
    assert "7.6.2" in section_ids
    block_762 = next(b for b in blocks if b["section_id"] == "7.6.2")
    assert "Locking of emergency locking retractors" in block_762["text"]
    assert "7.6.2.1" in block_762["text"]


def test_promotes_762_out_of_sibling_chapter_block():
    pages = [
        {
            "page_number": 20,
            "text": (
                "6.2.5.\n"
                "Retractor tests\n"
                "6.2.5.3.1.\n"
                "An emergency locking retractor, when tested in accordance with paragraph 7.6.2. below,\n"
                "7.6.2.\n"
                "Locking of emergency locking retractors\n"
                "7.6.2.1.\n"
                "The retractor shall be tested once.\n"
            ),
            "tables": [],
        }
    ]
    blocks = pages_to_structured_blocks(pages)
    assert any(b["section_id"] == "7.6.2" for b in blocks)
