"""Unit tests for structure-aware chunk metadata and enrichment."""

from __future__ import annotations

from ingestion.chunk import _parse_heading, _parent_section_number
from ingestion.enrich import context_header, enrich_chunk
from ingestion.models import Chunk


def test_parse_heading_unece_clause():
    num, title = _parse_heading("5.2.1. The Head Injury Criterion (HIC)")
    assert num == "5.2.1"
    assert "Head Injury" in title


def test_parse_heading_annex():
    num, title = _parse_heading("Annex 3 — Test procedure")
    assert num.lower().startswith("annex 3")
    assert "Test procedure" in title


def test_parent_section_number():
    assert _parent_section_number("5.2.1") == "5.2"
    assert _parent_section_number("5") is None


def test_enrich_header_format():
    chunk = Chunk(
        chunk_id="abc",
        text="The HIC shall not exceed 1000.",
        regulation_id="UN-ECE-R94",
        revision="Rev.3",
        section_number="5.2.1",
        section_title="Head Injury Criterion",
        section_id="UN-ECE-R94::5.2.1",
        parent_section_id="UN-ECE-R94::5.2",
        content_type="clause",
        page_number=12,
        bounding_box=[10.0, 20.0, 100.0, 40.0],
    )
    header = context_header(chunk)
    assert header.startswith("From UN-ECE-R94 Rev.3 §5.2.1")
    enriched = enrich_chunk(chunk)
    assert enriched.enriched_text.startswith(header)
    assert "HIC shall not exceed" in enriched.enriched_text


def test_chunk_payload_has_required_metadata():
    chunk = Chunk(
        chunk_id="t1",
        text="| A | B |\n|---|---|\n| 1 | 2 |",
        regulation_id="UN-ECE-R94",
        revision="Rev.3",
        section_number="5.3",
        section_title="Limits",
        section_id="UN-ECE-R94::table::1",
        parent_section_id="UN-ECE-R94::5.3",
        content_type="table",
        page_number=20,
        bounding_box=[1.0, 2.0, 3.0, 4.0],
    )
    payload = chunk.payload()
    for key in (
        "regulation_id",
        "revision",
        "section_number",
        "section_title",
        "page_number",
        "bounding_box",
        "content_type",
        "parent_section_id",
    ):
        assert key in payload
    assert payload["content_type"] == "table"
    assert payload["bounding_box"] == [1.0, 2.0, 3.0, 4.0]
