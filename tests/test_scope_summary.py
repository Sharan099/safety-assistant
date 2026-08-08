"""SCOPE_SUMMARY — hard-reg filter + deterministic sections + limits table."""

from __future__ import annotations

from generation.answer import AnswerSegment
from retrieval.router import QueryIntent, classify_query
from retrieval.scope_summary import (
    expand_scope_summary_query,
    format_limits_section,
    load_scope_summary_specs,
    render_scope_summary,
)
from retrieval.retrieve import RetrievedChunk
from ingestion.extract_limits import LimitRow


def test_router_scope_summary_r94():
    routed = classify_query("Summarize the scope of UN R94", use_llm=False, log=False)
    assert routed.intent == QueryIntent.SCOPE_SUMMARY
    assert routed.regulation_id == "UN-ECE-R94"


def test_expand_requires_named_reg():
    exp = expand_scope_summary_query("Summarize the scope of UN R94")
    assert exp.regulation_id == "UN-ECE-R94"
    assert exp.spec is not None
    assert exp.spec.scope_section == "1"


def test_specs_cover_indexed_regs():
    specs = load_scope_summary_specs(force=True)
    assert "UN-ECE-R94" in specs
    assert "UN-ECE-R95" in specs


def test_limits_section_uses_table_values():
    rows = [
        LimitRow(
            criterion_name="Thorax Compression Criterion",
            aliases=["ThCC"],
            limit_value=42.0,
            operator="<=",
            unit="mm",
            source_chunk_id="c1",
            section_number="5.2.1.4",
            regulation_id="UN-ECE-R94",
            verified=True,
        )
    ]
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="ThCC shall not exceed 42 mm",
        regulation_id="UN-ECE-R94",
        section_number="5.2.1.4",
        page_number=10,
        score=1.0,
    )
    text, sources = format_limits_section(
        rows, {"c1": chunk}, regulation_id="UN-ECE-R94", to_source=lambda c: c
    )
    assert "42" in text
    assert "Thorax Compression Criterion" in text
    assert len(sources) == 1


def test_render_rejects_foreign_chunks_in_segments():
    from retrieval.scope_summary import ScopeExpansion, ScopeRetrievalResult, ScopeSummarySpec

    spec = ScopeSummarySpec(
        regulation_id="UN-ECE-R94",
        label="UN R94",
        scope_section="1",
        definitions_section="2",
        key_requirement_sections=[],
        test_configuration_sections=[],
        homologation_sections=[],
    )
    r94 = RetrievedChunk(
        chunk_id="r94",
        text="This Regulation applies to vehicles of category M1.",
        regulation_id="UN-ECE-R94",
        section_number="1",
        page_number=1,
        score=1.0,
    )
    r95 = RetrievedChunk(
        chunk_id="r95",
        text="Side impact scope text.",
        regulation_id="UN-ECE-R95",
        section_number="1",
        page_number=1,
        score=1.0,
    )
    result = ScopeRetrievalResult(
        chunks=[r94],
        expansion=ScopeExpansion(
            question="Summarize the scope of UN R94",
            regulation_id="UN-ECE-R94",
            spec=spec,
        ),
        by_role={"scope": [r94]},
        chunk_role={"r94": "scope"},
        limits=[],
        missing_roles=[],
    )
    segs = [
        AnswerSegment(
            text="Applies to M1.",
            citation_chunk_id="r94",
            category_id="scope",
        ),
        AnswerSegment(
            text="LEAK from R95",
            citation_chunk_id="r95",
            category_id="scope",
        ),
    ]
    text, sources = render_scope_summary(
        result=result,
        segments=segs,
        chunks=[r94, r95],
        to_source=lambda c: c,
    )
    assert "LEAK from R95" not in text
    assert all(getattr(s, "regulation_id", "") == "UN-ECE-R94" for s in sources)
    assert "## Scope / applicability" in text
    assert "## Key injury criteria" in text
