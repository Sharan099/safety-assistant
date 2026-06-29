"""Tests for amendment front-matter vs filename audit (FR-16)."""

from __future__ import annotations

from registry.amendment_audit import audit_amendment_metadata


def test_amendment_match_no_mismatch():
    meta = {"amendment": "07 Series"}
    result = audit_amendment_metadata(meta, "UN_R14_07Series.pdf")
    assert result["amendment_mismatch"] is False


def test_amendment_mismatch_flagged():
    meta = {"amendment": "04 Series"}
    result = audit_amendment_metadata(meta, "UN_R14_07Series.pdf")
    assert result["amendment_mismatch"] is True
    assert result["amendment_from_filename"] == "07 Series"
