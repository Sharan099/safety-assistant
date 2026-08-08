"""Unit tests for preamble / annex heading merge / section helpers."""

from __future__ import annotations

from ingestion.chunk import (
    _is_numbered_section,
    _parse_heading,
    _parent_section_number,
)


def test_parse_annex_bare():
    num, title = _parse_heading("Annex 11")
    assert num.lower() == "annex 11"
    assert title == ""


def test_parse_annex_with_title():
    num, title = _parse_heading("Annex 6 — Procedure for determining the H-point")
    assert num.lower().startswith("annex 6")
    assert "H-point" in title or "Procedure" in title


def test_parse_clause():
    num, title = _parse_heading("5.2.1. Head Performance Criterion")
    assert num == "5.2.1"
    assert "Head" in title


def test_parent_annex_sub():
    assert _parent_section_number("Annex 11/1.2") == "Annex 11/1"
    assert _parent_section_number("Annex 11/1") == "Annex 11"


def test_numbered_rejects_preamble():
    assert not _is_numbered_section("Preamble")
    assert _is_numbered_section("Annex 11")
    assert _is_numbered_section("5.2.1")
