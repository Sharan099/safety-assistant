"""Tests for automatic Docling→LightOnOCR extraction validator."""

from __future__ import annotations

from types import SimpleNamespace

from docling_core.types.doc import DocItemLabel

from ingestion.extraction_validator import (
    clauses_are_plausible_neighbors,
    pages_needing_lighton,
    validate_extraction,
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


def _body(page: int, text: str) -> _FakeItem:
    return _FakeItem(DocItemLabel.TEXT, text, page)


def test_clause_in_caption_trigger():
    doc = _FakeDoc(
        [
            _FakeItem(DocItemLabel.PICTURE, "", 12),
            _FakeItem(DocItemLabel.CAPTION, "Figure 3", 12),
            _FakeItem(
                DocItemLabel.CAPTION,
                "5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;",
                12,
            ),
            _body(11, "5.2.1.6. The femur force criterion (FFC) shall not exceed 9.07 kN;\n" + ("x" * 500)),
            _body(13, "5.2.1.8. The tibia index (TI) shall not exceed 1.3;\n" + ("y" * 500)),
        ]
    )
    result = validate_extraction(doc, flag_density=False, flag_discontinuity=False)
    triggers = {f.trigger for f in result.flags if f.page_number == 12}
    assert "clause_in_caption" in triggers
    assert "figure_or_table" in triggers
    assert 12 in pages_needing_lighton(result)


def test_text_density_anomaly():
    items = []
    for p in range(1, 8):
        if p == 4:
            items.append(_body(p, "short"))  # dropped content
        else:
            items.append(_body(p, ("normal page content. " * 80) + f" clause {p}.1.1. body"))
    doc = _FakeDoc(items)
    result = validate_extraction(
        doc,
        flag_clause_in_caption=False,
        flag_figures_tables=False,
        flag_discontinuity=False,
    )
    dens = [f for f in result.flags if f.trigger == "text_density_anomaly"]
    assert any(f.page_number == 4 for f in dens)


def test_section_discontinuity_flags_both_pages():
    doc = _FakeDoc(
        [
            _body(10, "5.2.1.6. Femur force criterion text.\n" + ("a" * 500)),
            # Gap: 5.2.1.7 missing — jumps to 5.2.1.9
            _body(11, "5.2.1.9. Neck injury criterion text.\n" + ("b" * 500)),
        ]
    )
    result = validate_extraction(
        doc,
        flag_clause_in_caption=False,
        flag_figures_tables=False,
        flag_density=False,
    )
    disc = [f for f in result.flags if f.trigger == "section_discontinuity"]
    pages = {f.page_number for f in disc}
    assert pages == {10, 11}


def test_plausible_neighbors():
    assert clauses_are_plausible_neighbors("5.2.1.6", "5.2.1.7")
    assert clauses_are_plausible_neighbors("5.2.1.7", "5.2.2")
    assert not clauses_are_plausible_neighbors("5.2.1.6", "5.2.1.9")
    assert not clauses_are_plausible_neighbors("5.2.1.7", "5.2.1.5")
