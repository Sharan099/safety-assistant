"""Stage 2 gate: every figure has a non-empty description + valid parent clause."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.describe_figures import (
    PINNED_FIGURE_VLM_MODEL,
    FigureSpec,
    build_figure_chunks,
    caption_context_fallback,
    describe_and_chunk_figures,
    describe_figure,
    describe_figures,
    figures_from_stage1_extract,
    figures_from_vlm_result,
    pinned_figure_vlm_model,
)
from ingestion.extract import extract_from_docling, load_docling_json, remediate_clause_as_caption
from ingestion.vlm_figure_pass import (
    FigureBox,
    VlmFigurePassResult,
    VlmPageResult,
    parse_vlm_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
DOCLING_DIR = ROOT / "data" / "docling"

REGULATIONS = (
    ("UN_R94", "UN-ECE-R94", "Rev.3"),
    ("UN_R95", "UN-ECE-R95", "Rev.3"),
    ("UN_R16", "UN-ECE-R16", "Rev.10"),
    ("UN_R129", "UN-ECE-R129", "Rev.4"),
)

R94_P12 = """\
5.2.1.6. The femur force criterion (FFC) shall not exceed the force-time performance criterion shown in Figure 3;

Figure 3

**Femur force criterion**

![image](image_2.png)300,555,755,837

5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;
"""


def test_pinned_vlm_model_is_gemini_flash():
    assert pinned_figure_vlm_model() == PINNED_FIGURE_VLM_MODEL
    assert PINNED_FIGURE_VLM_MODEL == "gemini-2.5-flash"


def test_every_figure_has_nonempty_description_fallback():
    specs = [
        FigureSpec(
            page_number=12,
            figure_label="Figure 3",
            caption="Figure 3 — Femur force criterion",
            surrounding_text=(
                "5.2.1.6. The femur force criterion (FFC) shall not exceed the "
                "force-time performance criterion shown in Figure 3;"
            ),
            parent_section_number="5.2.1.6",
            bbox=[0.3, 0.5, 0.7, 0.8],
        ),
        FigureSpec(
            page_number=11,
            figure_label="Figure 1",
            caption="",
            surrounding_text="",
            parent_section_number="",
            bbox=[],
        ),
    ]
    descs = describe_figures(specs)
    assert len(descs) == 2
    for d in descs:
        assert d.searchable_text.strip()
        assert d.method in {"vlm", "caption_context_fallback"}
    # Femur curve gets heuristic enrichment in fallback.
    assert "force-time" in descs[0].searchable_text.lower()


def test_vlm_path_used_when_describe_fn_provided():
    fig = FigureSpec(
        page_number=12,
        figure_label="Figure 3",
        caption="Femur force criterion",
        surrounding_text="5.2.1.6. force-time performance criterion shown in Figure 3;",
        parent_section_number="5.2.1.6",
    )
    desc = describe_figure(
        fig,
        describe_fn=lambda _d: (
            "force-time performance curve for the femur force criterion, "
            "showing the maximum allowable force decreasing over the contact duration"
        ),
    )
    assert desc.method == "vlm"
    assert "decreasing over the contact duration" in desc.description


def test_figure_chunk_links_valid_parent_section():
    page = parse_vlm_markdown(R94_P12, page_number=12)
    vlm = VlmFigurePassResult(
        pages_input=[12],
        pages_processed=[12],
        pages_corrected=[],
        discrepancies=[],
        page_results={12: page},
    )
    specs = figures_from_vlm_result(vlm)
    assert specs
    assert specs[0].parent_section_number == "5.2.1.6"
    descs = describe_figures(specs)
    chunks = build_figure_chunks(descs, regulation_id="UN-ECE-R94", revision="Rev.3")
    assert chunks
    for ch in chunks:
        assert ch.content_type == "figure"
        assert ch.text.strip()
        assert ch.section_number == "5.2.1.6"
        assert ch.parent_section_id is not None
        assert "5.2.1.6" in (ch.parent_section_id or "")


def test_caption_context_fallback_never_empty():
    empty = FigureSpec(page_number=3)
    text = caption_context_fallback(empty)
    assert text.strip()
    assert "page 3" in text.lower()


def _extract_for(stem: str, regulation_id: str, revision: str):
    path = DOCLING_DIR / f"{stem}.docling.json"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    doc = load_docling_json(path)
    remediate_clause_as_caption(doc)
    return extract_from_docling(
        doc,
        document_id=regulation_id,
        regulation_id=regulation_id,
        revision=revision,
    )


@pytest.mark.parametrize("stem,regulation_id,revision", REGULATIONS)
def test_gate2_figure_chunks_nonempty_for_corpus(stem, regulation_id, revision):
    extract = _extract_for(stem, regulation_id, revision)
    specs = figures_from_stage1_extract(extract)
    if not specs:
        pytest.skip(f"{stem}: no figures in Stage 1 extract")
    chunks = describe_and_chunk_figures(
        regulation_id=regulation_id,
        revision=revision,
        extract=extract,
    )
    assert len(chunks) == len(specs)
    for ch in chunks:
        assert ch.content_type == "figure"
        assert ch.text.strip(), ch.section_id
        # Parent must be a real section number (clause) or explicit page fallback.
        assert ch.section_number
        assert ch.section_number.startswith("page-") or any(
            c.isdigit() for c in ch.section_number
        )


@pytest.mark.parametrize("stem,regulation_id,revision", REGULATIONS)
def test_gate2_figure_parent_resolves_to_section(stem, regulation_id, revision):
    extract = _extract_for(stem, regulation_id, revision)
    # Known section numbers present as headings/paragraphs in the extract.
    known_sections = {
        (e.section_number or "")
        for e in extract.elements
        if e.section_number
    }
    known_sections |= {
        m.group(1)
        for e in extract.elements
        for m in [__import__("re").match(r"^(\d+(?:\.\d+)*)\.\s+\S", e.text or "")]
        if m
    }
    chunks = describe_and_chunk_figures(
        regulation_id=regulation_id,
        revision=revision,
        extract=extract,
    )
    if not chunks:
        pytest.skip(f"{stem}: no figure chunks")
    linked = [c for c in chunks if c.parent_section_id and not c.section_number.startswith("page-")]
    # At least one figure should link to a known clause when the extract has sections.
    if known_sections and linked:
        assert any(
            c.section_number in known_sections
            or any(c.section_number in s or s in c.section_number for s in known_sections if s)
            for c in linked
        ), f"{stem}: no figure parent matched known sections"


def test_gate2_r94_figure3_parent_and_description():
    extract = _extract_for("UN_R94", "UN-ECE-R94", "Rev.3")
    # Prefer VLM-shaped spec for Figure 3 (authoritative parent).
    page = parse_vlm_markdown(R94_P12, page_number=12)
    vlm = VlmFigurePassResult(
        pages_input=[12],
        pages_processed=[12],
        pages_corrected=[],
        discrepancies=[],
        page_results={12: page},
    )
    chunks = describe_and_chunk_figures(
        regulation_id="UN-ECE-R94",
        revision="Rev.3",
        extract=extract,
        vlm_result=vlm,
    )
    fig3 = [c for c in chunks if "Figure 3" in c.text or "figure 3" in c.text.lower()]
    assert fig3, "Figure 3 chunk missing"
    assert all(c.text.strip() for c in fig3)
    assert any(c.section_number == "5.2.1.6" for c in fig3)
    assert any("force-time" in c.text.lower() or "femur" in c.text.lower() for c in fig3)
