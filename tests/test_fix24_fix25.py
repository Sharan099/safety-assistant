"""Fix 24 hard named-reg filter + Fix 25 plural multi-reg survey."""

from __future__ import annotations

from retrieval.enumerative import resolve_hard_regulation_filter
from retrieval.multi_regulation import is_plural_regulation_query
from retrieval.pipelines import overrides_from_routed
from retrieval.router import QueryIntent, classify_query, classify_regex
from retrieval.retrieve import retrieve


def test_fix24_named_reg_not_applicability():
    q = "What vehicles are covered under UN R94?"
    hit = classify_regex(q)
    assert hit is not None
    assert hit[0] == QueryIntent.FACTUAL_LOOKUP
    routed = classify_query(q, use_llm=False, log=False)
    assert routed.intent == QueryIntent.FACTUAL_LOOKUP
    assert routed.pipeline.hard_reg_filter is True
    assert resolve_hard_regulation_filter(q) == "UN-ECE-R94"


def test_fix24_retrieve_with_routed_never_cross_contaminates():
    q = "What vehicles are covered under UN R94?"
    routed = classify_query(q, use_llm=False, log=False)
    chunks = retrieve(
        q,
        top_k=5,
        rewrite=False,
        do_rerank=True,
        small_to_big=False,
        routed=routed,
    )
    assert chunks
    regs = {c.regulation_id for c in chunks}
    assert regs == {"UN-ECE-R94"}, regs


def test_fix24_comparative_still_skips_filter():
    assert resolve_hard_regulation_filter("How does UN R16 relate to UN R94?") is None
    assert resolve_hard_regulation_filter("compare R94 and R95") is None


def test_fix25_topic_plural_not_applicability():
    for q in (
        "Which regulations include electrical safety requirements?",
        "Which regulations cover electrical safety requirements?",
        "What electrical safety requirements apply in general?",
    ):
        assert is_plural_regulation_query(q), q
        hit = classify_regex(q)
        if hit is not None:
            assert hit[0] != QueryIntent.APPLICABILITY, (q, hit)
        routed = classify_query(q, use_llm=False, log=False)
        assert routed.intent != QueryIntent.APPLICABILITY, (q, routed.intent)


def test_fix25_vehicle_applicability_still_routes():
    q = "Which regulations apply to an M1 electric vehicle?"
    hit = classify_regex(q)
    assert hit is not None
    assert hit[0] == QueryIntent.APPLICABILITY
    ov = overrides_from_routed(classify_query(q, use_llm=False, log=False))
    assert ov.force_multi_reg is True
    assert ov.hard_reg_filter is False


def test_fix25_retrieve_reports_covered_and_missing_regs():
    from observability.context import reset_current_trace, set_current_trace
    from observability.trace import QueryTrace

    q = "Which regulations include electrical safety requirements?"
    tr = QueryTrace(trace_id="fix25-test", question=q)
    token = set_current_trace(tr)
    try:
        chunks = retrieve(q, top_k=3, rewrite=False, do_rerank=True, small_to_big=False)
        regs = {c.regulation_id for c in chunks}
        assert "UN-ECE-R94" in regs and "UN-ECE-R95" in regs, regs
        assert "UN-ECE-R16" not in regs
        covered = set(tr.optimizations.get("multi_regulation_covered") or [])
        missing = set(tr.optimizations.get("multi_regulation_missing") or [])
        assert "UN-ECE-R94" in covered and "UN-ECE-R95" in covered
        assert "UN-ECE-R16" in missing or "UN-ECE-R129" in missing
        assert tr.optimizations.get("multi_regulation") is True
    finally:
        reset_current_trace(token)
