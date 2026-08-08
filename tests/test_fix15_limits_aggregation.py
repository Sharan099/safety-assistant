"""Fix 15 enumerative breadth + limits-aggregation table + section bias."""

from __future__ import annotations

from retrieval.context_budget import apply_context_budget, context_budgets
from retrieval.enumerative import is_enumerative_query
from retrieval.limits_aggregation import (
    format_limits_markdown_table,
    is_limits_aggregation_query,
    render_limits_aggregation_answer,
)
from retrieval.router import QueryIntent, classify_query
from retrieval.section_bias import bias_chunks_by_section_category, detect_section_category
from retrieval.retrieve import RetrievedChunk


def test_limits_aggregation_detection():
    assert is_limits_aggregation_query(
        "Summarize all frontal impact injury limits"
    )
    assert is_limits_aggregation_query(
        "Summarize all pass/fail criteria for UN R94"
    )
    assert is_enumerative_query("Summarize all frontal impact injury limits")
    # Door dumps stay enumerative prose, not limits table.
    assert not is_limits_aggregation_query(
        "List every requirement related to doors in UN R95"
    )


def test_fix15_enum_budget_wins_over_factual_route():
    routed = classify_query(
        "Summarize all pass/fail criteria for UN R94", use_llm=False, log=False
    )
    # May be FACTUAL_LOOKUP; budget must still be broad.
    chunks, tokens, mode = context_budgets(
        "Summarize all pass/fail criteria for UN R94", routed=routed
    )
    assert chunks >= 15
    assert tokens >= 3000
    assert mode in {"enumerative", "limits_aggregation"}


def test_limits_table_render_covers_r94_criteria():
    from ingestion.extract_limits import load_limits_table, seed_known_limits

    seed_known_limits()
    table = load_limits_table("UN-ECE-R94")
    assert table and len(table.limits) >= 6
    md, _ids = format_limits_markdown_table(
        table.limits, regulation_id="UN-ECE-R94", chunks_by_id={}
    )
    assert "| Criterion |" in md
    lower = md.lower()
    for needle in ("hpc", "thcc", "viscous", "femur", "fuel", "isolation"):
        assert needle in lower, needle

    text, _sources = render_limits_aggregation_answer(
        "Summarize all frontal impact injury limits for UN R94",
        regulation_id="UN-ECE-R94",
        chunks=[],
    )
    assert "| Criterion |" in text
    assert text.count("|") >= 20  # multi-row table, not a paragraph


def test_section_category_bias_requirements():
    assert detect_section_category("What are the installation requirements?") in {
        "installation",
        "requirements",
    }
    assert detect_section_category("Define H-point") == "definitions"
    assert detect_section_category("What is the scope of UN R94?") == "scope"

    chunks = [
        RetrievedChunk(
            chunk_id="scope",
            text="Scope of the regulation",
            section_number="1",
            section_title="Scope",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="req",
            text="Performance requirements shall not exceed",
            section_number="5.2.1",
            section_title="Specifications",
            score=0.4,
        ),
    ]
    ordered = bias_chunks_by_section_category(
        chunks, "Summarize all performance requirements in UN R94", category="requirements"
    )
    assert ordered[0].chunk_id == "req"
