"""Classifier scoring tests — deterministic, no I/O."""

from backend.app.gateway.classifier import classify, score_only
from backend.app.gateway.types import RoutingContext


def test_simple_grounded_query_routes_tier1():
    ctx = RoutingContext(
        prompt="What is UN R14?",
        query="What is UN R14?",
        grounding={"confidence": 0.92, "best_semantic": 0.92},
        conversation_depth=0,
    )
    d = classify(ctx)
    assert d.tier == 1
    assert d.provider == "groq"
    assert 0.0 <= d.score <= 10.0


def test_code_and_reasoning_routes_high_tier():
    ctx = RoutingContext(
        prompt=(
            "Debug this crash analysis and explain why the seat anchorage FE model "
            "diverges. ```python\nfor i in range(n):\n    solve(i)\n``` "
            "Compare strategies and propose a multi-step plan."
        ),
        query="debug FE model",
        grounding={"confidence": 0.4, "best_semantic": 0.4},
        conversation_depth=5,
    )
    d = classify(ctx)
    assert d.tier == 3
    assert d.provider == "anthropic_sonnet"
    assert any("reasoning" in r or "code" in r for r in d.reasons)


def test_low_grounding_increases_score():
    base = RoutingContext(prompt="Summarise the test load requirements.",
                          query="x", grounding={"confidence": 0.9})
    low = RoutingContext(prompt="Summarise the test load requirements.",
                         query="x", grounding={"confidence": 0.1})
    assert classify(low).score >= classify(base).score


def test_score_only_shape():
    out = score_only(RoutingContext(prompt="hello", query="hello"))
    assert set(out.keys()) == {"score", "reasons"}
    assert isinstance(out["score"], float)
    assert isinstance(out["reasons"], list)


def test_feedback_history_escalates():
    happy = RoutingContext(prompt="compare R14 and R16 belt loads",
                           query="x", feedback_downvote_rate=0.0)
    unhappy = RoutingContext(prompt="compare R14 and R16 belt loads",
                             query="x", feedback_downvote_rate=0.9)
    assert classify(unhappy).score >= classify(happy).score
