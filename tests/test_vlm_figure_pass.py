"""Unit tests for selective VLM figure pass (no model download)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from docling_core.types.doc import DocItemLabel

from ingestion.vlm_figure_pass import (
    apply_vlm_preference_to_page,
    load_audit_page_list,
    parse_vlm_markdown,
    reading_order_disagrees,
)


def test_parse_vlm_markdown_figures_and_clauses():
    md = """\
5.2.1.6. The femur force criterion (FFC) shall not exceed the force-time performance criterion shown in Figure 3;

Figure 3

**Femur force criterion**

![image](image_2.png)300,555,755,837

5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;
"""
    result = parse_vlm_markdown(md, page_number=12)
    assert result.clause_sequence == ["5.2.1.6", "5.2.1.7"]
    assert len(result.figures) == 1
    assert result.figures[0].bbox_norm == (300, 555, 755, 837)
    assert result.figures[0].parent_section_number == "5.2.1.6"
    assert "Figure 3" in result.figures[0].caption or "Femur" in result.figures[0].caption


def test_reading_order_disagrees_on_caption_mislabel():
    disagrees, reason = reading_order_disagrees(
        docling_clauses=["5.2.1.3", "5.2.1.6", "5.2.1.7"],
        vlm_clauses=["5.2.1.3", "5.2.1.6", "5.2.1.7"],
        caption_mislabels=["5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;"],
    )
    assert disagrees is True
    assert reason == "docling_caption_is_clause"


def test_reading_order_keeps_docling_when_vlm_empty():
    disagrees, reason = reading_order_disagrees(
        docling_clauses=["5.2.1.3"],
        vlm_clauses=[],
        caption_mislabels=[],
    )
    assert disagrees is False
    assert reason == "vlm_empty_keep_docling"


def test_load_audit_json(tmp_path: Path):
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "figure_adjacent_pages": [11, 12, 13, 20],
                "suspect_pages": [12],
            }
        ),
        encoding="utf-8",
    )
    assert load_audit_page_list(path) == [11, 12, 13, 20]


def test_load_audit_legacy_txt(tmp_path: Path):
    path = tmp_path / "audit.txt"
    path.write_text(
        "Figure/table-adjacent suspects: 1\n"
        "  [figure_adjacent_page] section='Annex 1' text_clause='14' page=20 id=abc\n"
        "  section='Annex 4' text_clause='1.1' page=24 id=def\n",
        encoding="utf-8",
    )
    assert load_audit_page_list(path) == [20, 24]


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


def test_apply_vlm_relabels_caption_clause():
    """Stage 1 bug: Docling tags 5.2.1.7 as caption after Figure 3."""
    caption_clause = (
        "5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;"
    )
    items = [
        _FakeItem(DocItemLabel.LIST_ITEM, "5.2.1.6. The femur force criterion … Figure 3;", 12),
        _FakeItem(DocItemLabel.CAPTION, "Figure 3", 12),
        _FakeItem(DocItemLabel.PICTURE, "", 12),
        _FakeItem(DocItemLabel.CAPTION, caption_clause, 12),
    ]
    doc = _FakeDoc(items)
    vlm = parse_vlm_markdown(
        "5.2.1.6. The femur force criterion … Figure 3;\n\n"
        "Figure 3\n\n**Femur force criterion**\n\n"
        "![image](image_1.png)1,2,3,4\n\n"
        f"{caption_clause}\n",
        page_number=12,
    )
    actions = apply_vlm_preference_to_page(doc, 12, vlm)
    assert any(a.startswith("relabel_caption_to_text:") for a in actions)
    assert items[-1].label == DocItemLabel.TEXT
    assert items[-1].text.startswith("5.2.1.7.")
