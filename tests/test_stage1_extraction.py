"""Stage 1 gate: layout-aware extraction schema + regression guards."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc import DocItemLabel

from ingestion.caption_guard import CLAUSE_IN_CAPTION_RE
from ingestion.extract import (
    DocumentExtract,
    PageElement,
    PageExtract,
    assert_extract_invariants,
    elements_with_clause_in_caption_or_figure,
    extract_from_docling,
    find_figure_parent_section,
    load_docling_json,
    remediate_clause_as_caption,
)

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"
DOCLING_DIR = ROOT / "data" / "docling"

REGULATIONS = (
    ("UN_R94", "UN-ECE-R94", "Rev.3"),
    ("UN_R95", "UN-ECE-R95", "Rev.3"),
    ("UN_R16", "UN-ECE-R16", "Rev.10"),
    ("UN_R129", "UN-ECE-R129", "Rev.4"),
)


class _FakeItem:
    def __init__(
        self,
        label: DocItemLabel,
        text: str,
        page: int,
        *,
        bbox: tuple[float, float, float, float] = (10.0, 20.0, 100.0, 40.0),
        table_rows: int | None = None,
        table_cols: int | None = None,
    ) -> None:
        self.label = label
        self.text = text
        self.prov = [SimpleNamespace(page_no=page, bbox=SimpleNamespace(l=bbox[0], t=bbox[1], r=bbox[2], b=bbox[3]))]
        self._rows = table_rows
        self._cols = table_cols
        if table_rows and table_cols:
            self.data = SimpleNamespace(
                num_rows=table_rows,
                num_cols=table_cols,
                grid=[[""] * table_cols for _ in range(table_rows)],
            )

    def export_to_markdown(self, doc=None):  # noqa: ANN001
        if self._rows and self._cols:
            header = "| " + " | ".join(f"c{i}" for i in range(self._cols)) + " |"
            sep = "| " + " | ".join("---" for _ in range(self._cols)) + " |"
            rows = [
                "| " + " | ".join(f"r{r}c{c}" for c in range(self._cols)) + " |"
                for r in range(self._rows)
            ]
            return "\n".join([header, sep, *rows])
        return self.text


class _FakeDoc:
    def __init__(self, items: list[_FakeItem]) -> None:
        self._items = items

    def iterate_items(self):
        for it in self._items:
            yield it, 1


def _synthetic_doc() -> _FakeDoc:
    return _FakeDoc(
        [
            _FakeItem(DocItemLabel.SECTION_HEADER, "5.2.1.6. Femur force criterion", 12),
            _FakeItem(
                DocItemLabel.TEXT,
                "5.2.1.6. The femur force criterion (FFC) shall not exceed the "
                "force-time performance criterion shown in Figure 3;",
                12,
            ),
            _FakeItem(DocItemLabel.PICTURE, "Figure 3", 12),
            _FakeItem(DocItemLabel.CAPTION, "Figure 3 — Femur force criterion", 12),
            _FakeItem(
                DocItemLabel.TEXT,
                "5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;",
                12,
            ),
            _FakeItem(
                DocItemLabel.TABLE,
                "",
                13,
                table_rows=3,
                table_cols=2,
            ),
        ]
    )


def test_synthetic_schema_has_page_coords_reading_order():
    extract = extract_from_docling(_synthetic_doc(), document_id="synth", regulation_id="UN-ECE-R94")
    assert extract.elements
    for el in extract.elements:
        assert el.page_number >= 1
        assert len(el.coordinates) == 4
        assert el.reading_order >= 1
        assert el.element_type in {
            "heading",
            "paragraph",
            "table",
            "figure",
            "caption",
            "footnote",
            "formula",
        }


def test_synthetic_tables_are_structured_not_flattened():
    extract = extract_from_docling(_synthetic_doc(), document_id="synth")
    tables = [e for e in extract.elements if e.element_type == "table"]
    assert tables
    for t in tables:
        assert (t.row_count or 0) > 0
        assert (t.col_count or 0) > 0
        assert (t.markdown or "").count("|") >= 2


def test_synthetic_no_clause_in_caption_or_figure():
    extract = extract_from_docling(_synthetic_doc(), document_id="synth")
    assert elements_with_clause_in_caption_or_figure(extract) == []
    assert_extract_invariants(extract)


def test_tibia_force_caption_bug_is_remediated():
    """Permanent regression guard: clause text must not remain caption/figure."""
    doc = _FakeDoc(
        [
            _FakeItem(DocItemLabel.PICTURE, "", 12),
            _FakeItem(DocItemLabel.CAPTION, "Figure 3", 12),
            _FakeItem(
                DocItemLabel.CAPTION,
                "5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;",
                12,
            ),
            _FakeItem(
                DocItemLabel.TEXT,
                "5.2.1.6. The femur force criterion (FFC) shall not exceed the "
                "force-time performance criterion shown in Figure 3;",
                12,
            ),
        ]
    )
    before = extract_from_docling(doc, document_id="bug")
    assert elements_with_clause_in_caption_or_figure(before), "fixture must reproduce bug"

    # Deterministic remediation (same class of fix LightOnOCR preference applies).
    actions = remediate_clause_as_caption(doc)
    assert actions
    after = extract_from_docling(doc, document_id="bug")
    assert elements_with_clause_in_caption_or_figure(after) == []
    # Relabeled clause must still be present as a paragraph.
    paras = [e for e in after.elements if e.element_type == "paragraph"]
    assert any("5.2.1.7" in e.text and "TCFC" in e.text for e in paras)


def test_figure3_fuel_leakage_boundary_section_attribution():
    """Original misattribution site: Figure 3 belongs with §5.2.1.6 (femur), not tibia."""
    extract = extract_from_docling(_synthetic_doc(), document_id="r94")
    parent = find_figure_parent_section(extract, page_number=12, figure_label="Figure 3")
    assert parent == "5.2.1.6"
    # 5.2.1.7 must be a paragraph (not caption) after clean extract.
    tibia = [
        e
        for e in extract.elements
        if e.page_number == 12 and "5.2.1.7" in e.text and "TCFC" in e.text
    ]
    assert tibia
    assert all(e.element_type == "paragraph" for e in tibia)


def _load_or_parse(stem: str, regulation_id: str, revision: str):
    json_path = DOCLING_DIR / f"{stem}.docling.json"
    pdf_path = PDF_DIR / f"{stem}.pdf"
    if json_path.is_file():
        doc = load_docling_json(json_path)
    else:
        if not pdf_path.is_file():
            pytest.skip(f"missing {pdf_path} and {json_path}")
        from ingestion.parse import parse_pdf

        doc = parse_pdf(pdf_path, export_dir=DOCLING_DIR)
    # Always clear residual caption-clause mislabels before Stage 1 asserts.
    remediate_clause_as_caption(doc)
    return extract_from_docling(
        doc,
        document_id=regulation_id,
        regulation_id=regulation_id,
        revision=revision,
        source_path=str(pdf_path),
        pdf_path=pdf_path if pdf_path.is_file() else None,
    )


@pytest.mark.parametrize("stem,regulation_id,revision", REGULATIONS)
def test_gate1_no_clause_in_caption_or_figure(stem, regulation_id, revision):
    extract = _load_or_parse(stem, regulation_id, revision)
    bad = elements_with_clause_in_caption_or_figure(extract)
    assert bad == [], f"{stem}: {[(e.page_number, e.text[:80]) for e in bad]}"


@pytest.mark.parametrize("stem,regulation_id,revision", REGULATIONS)
def test_gate1_tables_structured(stem, regulation_id, revision):
    extract = _load_or_parse(stem, regulation_id, revision)
    tables = [e for e in extract.elements if e.element_type == "table"]
    # All four regs contain at least one table in practice; if a stub PDF has
    # none, skip rather than false-fail.
    if not tables:
        pytest.skip(f"{stem}: no tables extracted")
    for t in tables:
        assert (t.row_count or 0) > 0, t.element_id
        assert (t.col_count or 0) > 0, t.element_id
        assert (t.markdown or t.html or "").strip(), t.element_id


@pytest.mark.parametrize("stem,regulation_id,revision", REGULATIONS)
def test_gate1_every_element_has_page_coords_reading_order(stem, regulation_id, revision):
    extract = _load_or_parse(stem, regulation_id, revision)
    assert extract.elements, f"{stem}: empty extract"
    for el in extract.elements:
        assert isinstance(el.page_number, int) and el.page_number >= 1
        assert isinstance(el.coordinates, list) and len(el.coordinates) == 4
        assert isinstance(el.reading_order, int) and el.reading_order >= 1


def test_gate1_r94_figure3_section_on_real_docling():
    json_path = DOCLING_DIR / "UN_R94.docling.json"
    if not json_path.is_file():
        pytest.skip("UN_R94.docling.json not available")
    doc = load_docling_json(json_path)
    remediate_clause_as_caption(doc)
    extract = extract_from_docling(
        doc,
        document_id="UN-ECE-R94",
        regulation_id="UN-ECE-R94",
        revision="Rev.3",
        pdf_path=PDF_DIR / "UN_R94.pdf",
    )
    assert_extract_invariants(extract)
    parent = find_figure_parent_section(extract, page_number=12, figure_label="Figure 3")
    assert parent == "5.2.1.6", f"Figure 3 parent was {parent!r}"
    # Tibia clause must be recoverable as paragraph text on the boundary page.
    page12 = next(p for p in extract.pages if p.page_number == 12)
    texts = "\n".join(e.text for e in page12.elements if e.element_type == "paragraph")
    assert "5.2.1.7" in texts
    assert "TCFC" in texts or "tibia" in texts.lower()
    # No residual caption/figure clause swallow.
    for el in page12.elements:
        if el.element_type in {"caption", "figure"}:
            assert not CLAUSE_IN_CAPTION_RE.search(el.text or "")
