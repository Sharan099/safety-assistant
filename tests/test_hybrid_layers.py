"""Hybrid Layer 1–2 vs 3–5 routing (fast vs bounded multi-step)."""

from __future__ import annotations

from agent.planner import plan_for_query_intent
from generation.llm_client import LLMClient
from retrieval.hybrid_layers import (
    ExecutionLayer,
    FAST_PATH_INTENTS,
    MULTI_STEP_INTENTS,
    execution_layer_for,
    should_use_multi_step_layer,
)
from retrieval.router import QueryIntent, classify_query


def test_hybrid_layer_sets():
    assert QueryIntent.FACTUAL_LOOKUP in FAST_PATH_INTENTS
    assert QueryIntent.COMPLIANCE_CHECK in FAST_PATH_INTENTS
    assert QueryIntent.DESIGN_IMPLICATION in MULTI_STEP_INTENTS
    assert QueryIntent.APPLICABILITY in MULTI_STEP_INTENTS
    assert QueryIntent.CHECKLIST_GEN in MULTI_STEP_INTENTS
    assert QueryIntent.RETEST_SCOPE in MULTI_STEP_INTENTS
    assert QueryIntent.SCOPE_SUMMARY not in MULTI_STEP_INTENTS


def test_fast_path_intents_not_multi_step():
    for q, intent in (
        ("What is the HPC limit in UN R94?", QueryIntent.FACTUAL_LOOKUP),
        (
            "HPC measured 850 — does the vehicle pass UN R94?",
            QueryIntent.COMPLIANCE_CHECK,
        ),
    ):
        routed = classify_query(q, use_llm=False, log=False)
        assert routed.intent == intent, (q, routed.intent)
        assert execution_layer_for(routed) is ExecutionLayer.FAST
        assert not should_use_multi_step_layer(routed)


def test_multi_step_intents_enter_layer_35():
    cases = (
        ("What requirements affect B-Pillar design?", QueryIntent.DESIGN_IMPLICATION),
        (
            "Which regulations apply to an M1 electric vehicle?",
            QueryIntent.APPLICABILITY,
        ),
        (
            "Generate a checklist for preparing a vehicle for UN R94 homologation",
            QueryIntent.CHECKLIST_GEN,
        ),
        (
            "After changing the REESS mounting, do we need to retest?",
            QueryIntent.RETEST_SCOPE,
        ),
    )
    for q, intent in cases:
        routed = classify_query(q, use_llm=False, log=False)
        assert routed.intent == intent, (q, routed.intent)
        assert execution_layer_for(routed) is ExecutionLayer.MULTI_STEP
        assert should_use_multi_step_layer(routed)
        mode, plan = plan_for_query_intent(intent.value, q)
        assert len(plan) == 1
        assert plan[0].tool in {
            "design_implication",
            "applicability",
            "checklist_gen",
            "retest_scope",
        }
        assert mode == plan[0].tool


def test_agent_fast_path_short_circuits_factual():
    from agent.loop import run_agent

    result = run_agent(
        "What is the HPC limit in UN R94?",
        llm=LLMClient(provider="mock", use_cache=False),
    )
    assert result.mode == "fast_path"
    assert result.execution_layer == "fast"
    assert result.multi_step is False
    assert result.query_intent == "FACTUAL_LOOKUP"


def test_agent_multi_step_plans_design():
    from agent.loop import run_agent

    result = run_agent(
        "What requirements affect B-Pillar design?",
        llm=LLMClient(provider="mock", use_cache=False),
    )
    assert result.mode == "design_implication"
    assert result.execution_layer == "multi_step"
    assert result.multi_step is True
    assert any(s.tool == "design_implication" for s in result.steps)
