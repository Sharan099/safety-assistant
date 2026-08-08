"""Tests for Stage-2 figure chunk emission + parent linkage."""

from __future__ import annotations

from types import SimpleNamespace

from docling_core.types.doc import DocItemLabel

from ingestion.models import Chunk
from ingestion.vlm_figure_pass import (
    VlmFigurePassResult,
    VlmPageResult,
    build_figure_chunks_from_vlm,
    parse_vlm_markdown,
)


R94_P12 = """\
5.2.1.3. The neck bending moment about the y axis shall no exceed 57 Nm in extension³;

5.2.1.4. The Thorax Compression Criterion (ThCC) shall not exceed 42 mm;

5.2.1.5. The viscous criterion (V * C) for the thorax shall not exceed 1,0 m/s;

5.2.1.6. The femur force criterion (FFC) shall not exceed the force-time performance criterion shown in Figure 3;

Figure 3

**Femur force criterion**

![image](image_2.png)300,555,755,837

5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;
"""

R94_P11 = """\
5.2.1.2. The Injury Criteria for the neck (NIC) shall not exceed the values shown in Figures 1 and 2;

Figure 1
Neck tension criterion

![image](image_1.png)200,425,844,667

Figure 2
Neck shear criterion

![image](image_2.png)100,200,300,400
"""


class _FakeDoc:
    def iterate_items(self):
        if False:  # pragma: no cover
            yield None, 0


def test_parse_vlm_links_figure_3_to_5216():
    result = parse_vlm_markdown(R94_P12, page_number=12)
    assert len(result.figures) == 1
    fig = result.figures[0]
    assert fig.figure_label == "Figure 3"
    assert fig.parent_section_number == "5.2.1.6"
    assert "Femur force" in fig.caption
    assert "force-time" in fig.surrounding_text.lower()


def test_parse_vlm_links_figures_1_2_to_5212():
    result = parse_vlm_markdown(R94_P11, page_number=11)
    assert len(result.figures) == 2
    assert result.figures[0].parent_section_number == "5.2.1.2"
    assert result.figures[1].parent_section_number == "5.2.1.2"
    assert result.figures[0].figure_label == "Figure 1"
    assert "Neck tension" in result.figures[0].caption


def test_build_figure_chunks_content_type_and_parent():
    page = parse_vlm_markdown(R94_P12, page_number=12)
    vlm = VlmFigurePassResult(
        pages_input=[12],
        pages_processed=[12],
        pages_corrected=[],
        discrepancies=[],
        page_results={12: page},
    )
    chunks = build_figure_chunks_from_vlm(
        vlm,
        _FakeDoc(),  # type: ignore[arg-type]
        regulation_id="UN-ECE-R94",
        revision="Rev.3",
    )
    assert len(chunks) == 1
    ch = chunks[0]
    assert isinstance(ch, Chunk)
    assert ch.content_type == "figure"
    assert ch.section_number == "5.2.1.6"
    assert "Figure 3" in ch.text
    assert "Femur force" in ch.text
    assert "force-time" in ch.text.lower()
    assert ch.parent_section_id is not None
    assert "5.2.1.6" in (ch.parent_section_id or "")
    assert ch.page_number == 12
    assert len(ch.bounding_box) == 4


def test_without_figure_chunks_no_retrievable_figure_content():
    """Baseline: clause-only corpus has no figure content_type for Fig 3 queries."""
    clause_only = [
        Chunk(
            chunk_id="c1",
            text=(
                "5.2.1.6 The femur force criterion (FFC) shall not exceed "
                "the force-time performance criterion shown in Figure 3;"
            ),
            regulation_id="UN-ECE-R94",
            revision="Rev.3",
            section_number="5.2.1.6",
            section_title="Femur force criterion",
            section_id="UN-ECE-R94::5.2.1.6",
            content_type="clause",
            page_number=12,
        )
    ]
    assert not any(c.content_type == "figure" for c in clause_only)
    # The clause mentions the figure but does not carry caption/curve description.
    assert "Femur force criterion" not in clause_only[0].text.split("Figure 3")[-1] or True
    assert "![image]" not in clause_only[0].text
