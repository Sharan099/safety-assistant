"""Query intent router — regex fast-paths, per-intent budgets, audit log."""

from __future__ import annotations

import json
from pathlib import Path

from retrieval.context_budget import apply_context_budget, context_budgets
from retrieval.pipelines import overrides_from_routed
from retrieval.retrieve import RetrievedChunk
from retrieval.router import (
    PIPELINE_CONFIGS,
    QueryIntent,
    RetrievalStrategy,
    classify_query,
    classify_regex,
    pipeline_for,
)


def test_regex_routes_core_intents():
    cases = [
        ("What is the HPC limit in UN R94?", QueryIntent.FACTUAL_LOOKUP),
        (
            "HPC measured 850 — does the vehicle pass UN R94?",
            QueryIntent.COMPLIANCE_CHECK,
        ),
        (
            "What requirements affect the side door structure design?",
            QueryIntent.DESIGN_IMPLICATION,
        ),
        (
            "Generate a checklist for preparing a vehicle for UN R94 homologation",
            QueryIntent.CHECKLIST_GEN,
        ),
        (
            "Summarize the scope of UN R95",
            QueryIntent.SCOPE_SUMMARY,
        ),
        (
            "Which regulations apply to an M1 electric vehicle?",
            QueryIntent.APPLICABILITY,
        ),
        (
            "After changing the REESS mounting, do we need to retest?",
            QueryIntent.RETEST_SCOPE,
        ),
        # Fix 24: named-reg coverage is factual, not corpus applicability
        (
            "What vehicles are covered under UN R94?",
            QueryIntent.FACTUAL_LOOKUP,
        ),
        # Fix 25: topic plural survey is not vehicle APPLICABILITY
        (
            "Which regulations include electrical safety requirements?",
            QueryIntent.FACTUAL_LOOKUP,  # default/factual; plural retrieve handles survey
        ),
    ]
    for q, expected in cases:
        hit = classify_regex(q)
        # Topic plural may return None (falls to default FACTUAL) or FACTUAL cue
        if expected == QueryIntent.FACTUAL_LOOKUP and hit is None:
            routed = classify_query(q, use_llm=False, log=False)
            assert routed.intent == QueryIntent.FACTUAL_LOOKUP, q
            continue
        assert hit is not None, q
        intent, _reason = hit
        assert intent == expected, (q, intent, expected)
        assert intent != QueryIntent.APPLICABILITY or "apply to" in q.lower()


def test_classify_query_defaults_without_llm(tmp_path: Path, monkeypatch):
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("QUERY_INTENT_AUDIT_PATH", str(audit))
    monkeypatch.setenv("QUERY_ROUTER_LLM", "0")

    routed = classify_query(
        "Tell me something about frontal impact protection",
        use_llm=False,
        log=True,
    )
    assert routed.intent == QueryIntent.FACTUAL_LOOKUP
    assert routed.source in {"default", "regex"}
    assert audit.exists()
    lines = audit.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["intent"] == "FACTUAL_LOOKUP"
    assert rec["retrieval_strategy"]
    assert rec["budget_mode"]


def test_pipeline_configs_are_distinct():
    modes = {p.budget_mode for p in PIPELINE_CONFIGS.values()}
    assert len(modes) == len(QueryIntent)
    strategies = {p.retrieval_strategy for p in PIPELINE_CONFIGS.values()}
    assert RetrievalStrategy.STANDARD in strategies
    assert RetrievalStrategy.MULTI_REG_TOPIC in strategies
    assert RetrievalStrategy.CHECKLIST in strategies
    design = pipeline_for(QueryIntent.DESIGN_IMPLICATION)
    factual = pipeline_for(QueryIntent.FACTUAL_LOOKUP)
    assert design.max_chunks > factual.max_chunks
    assert design.multi_reg_loop is True
    assert factual.multi_reg_loop is False
    assert design.hard_reg_filter is False
    assert factual.hard_reg_filter is True


def test_routed_budgets_override_heuristics():
    routed = classify_query(
        "What requirements affect seat anchorage design?",
        use_llm=False,
        log=False,
    )
    assert routed.intent == QueryIntent.DESIGN_IMPLICATION
    chunks, tokens, mode = context_budgets("x", routed=routed)
    assert mode == "design_implication"
    assert chunks == routed.pipeline.max_chunks
    assert tokens == routed.pipeline.max_tokens

    sample = [
        RetrievedChunk(chunk_id=f"c{i}", text="word " * 20, score=0.5)
        for i in range(30)
    ]
    trimmed, stats = apply_context_budget(
        sample,
        question="What requirements affect seat anchorage design?",
        routed=routed,
    )
    assert stats["mode"] == "design_implication"
    assert len(trimmed) <= routed.pipeline.max_chunks


def test_retrieve_overrides_from_routed():
    routed = classify_query(
        "Generate a checklist for preparing a vehicle for UN R94 testing",
        use_llm=False,
        log=False,
    )
    assert routed.intent == QueryIntent.CHECKLIST_GEN
    ov = overrides_from_routed(routed)
    assert ov is not None
    assert ov.force_enumerative is True
    assert ov.rerank_top_k >= 10
    assert ov.hard_reg_filter is True

    design = classify_query(
        "What requirements affect the door design?",
        use_llm=False,
        log=False,
    )
    dov = overrides_from_routed(design)
    assert dov is not None
    assert dov.force_multi_reg is True
    assert dov.hard_reg_filter is False


def test_retest_beats_compliance_when_both_cued():
    hit = classify_regex(
        "After changing the bumper, does the vehicle still pass — do we need to retest?"
    )
    assert hit is not None
    assert hit[0] == QueryIntent.RETEST_SCOPE


def test_enumerative_list_maps_to_checklist():
    routed = classify_query(
        "List every requirement related to doors in UN R95",
        use_llm=False,
        log=False,
    )
    assert routed.intent == QueryIntent.CHECKLIST_GEN
