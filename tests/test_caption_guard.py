"""Tests for permanent clause-in-caption ingestion guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from docling_core.types.doc import DocItemLabel

from ingestion.caption_guard import (
    CaptionClauseGuardError,
    find_caption_clause_violations,
    find_clause_numbers_in_text,
    flagged_pages,
    validate_no_clause_as_caption,
)


class _FakeItem:
    def __init__(self, label: DocItemLabel, text: str, page: int) -> None:
        self.label = label
        self.text = text
        self.prov = [SimpleNamespace(page_no=page, bbox=None)]


class _FakeDoc:
    def __init__(self, items: list[_FakeItem]) -> None:
        self._items = items

    def iterate_items(self):
        for it in self._items:
            yield it, 1


def test_clause_pattern_requires_three_segments():
    assert find_clause_numbers_in_text("Figure 3 — Femur force") == []
    assert find_clause_numbers_in_text("5.2.1.7. The tibia") == ["5.2.1.7"]
    assert find_clause_numbers_in_text("6.3.5.1 Support-leg") == ["6.3.5.1"]
    assert find_clause_numbers_in_text("see 1.2 only") == []  # only two segments


def test_detects_tcfc_caption_mislabel():
    caption = (
        "5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;"
    )
    doc = _FakeDoc(
        [
            _FakeItem(DocItemLabel.CAPTION, "Figure 3", 12),
            _FakeItem(DocItemLabel.CAPTION, caption, 12),
            _FakeItem(DocItemLabel.TEXT, "unrelated body", 12),
        ]
    )
    hits = find_caption_clause_violations(doc)
    assert len(hits) == 1
    assert hits[0].page_number == 12
    assert "5.2.1.7" in hits[0].clause_numbers
    assert flagged_pages(hits) == [12]


def test_ignores_real_figure_captions():
    doc = _FakeDoc(
        [
            _FakeItem(DocItemLabel.CAPTION, "Figure 3 — Femur force criterion", 12),
            _FakeItem(
                DocItemLabel.TEXT,
                "5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;",
                12,
            ),
        ]
    )
    assert find_caption_clause_violations(doc) == []


def test_validate_raises_when_strict():
    doc = _FakeDoc(
        [
            _FakeItem(
                DocItemLabel.CAPTION,
                "6.3.5.1. Support-leg and support-leg foot geometrical requirements",
                31,
            )
        ]
    )
    with pytest.raises(CaptionClauseGuardError):
        validate_no_clause_as_caption(doc, raise_on_violation=True)
