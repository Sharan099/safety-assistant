"""
Complexity classifier — the routing scoring engine.

Pure and deterministic: given a RoutingContext it returns a RouteDecision with a
0-10 score, human-readable reasons, and the selected tier/provider/model. No I/O,
no network, no global state -> trivially unit-testable.

Each of the 10 routing inputs is normalised to 0..1, multiplied by its weight,
summed, and scaled to 0..10. The grounding/retrieval confidence signals are
REUSED from the existing grounding assessment (no recomputation).
"""

from __future__ import annotations

import re

from backend.app.gateway import config as cfg
from backend.app.gateway.types import RouteDecision, RoutingContext

_WORD_RE = re.compile(r"\S+")
_CODE_FENCE_RE = re.compile(r"```|~~~|\b\w+\([^)]*\)\s*[{:]")
_MULTI_CLAUSE_RE = re.compile(r"[;,]|\band\b|\bor\b|\bvs\.?\b|\bversus\b", re.I)
_QUESTION_RE = re.compile(r"\?")


def _estimate_tokens(text: str) -> int:
    # Mirrors metrics.estimate_tokens heuristic (word count * 1.33).
    return max(1, int(len(_WORD_RE.findall(text)) * 1.33))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _detect_code(text: str) -> bool:
    low = text.lower()
    if _CODE_FENCE_RE.search(text):
        return True
    return any(kw in low for kw in cfg.CODE_KEYWORDS)


def _reasoning_hits(text: str) -> list[str]:
    low = text.lower()
    return [kw for kw in cfg.REASONING_KEYWORDS if kw in low]


def _query_complexity(text: str) -> float:
    """Structural/lexical complexity 0..1 from length, clauses and questions."""
    words = len(_WORD_RE.findall(text))
    clause_signals = len(_MULTI_CLAUSE_RE.findall(text))
    questions = len(_QUESTION_RE.findall(text))
    # Long, multi-clause, multi-question prompts are more complex.
    length_term = _clamp01(words / 120.0)
    clause_term = _clamp01(clause_signals / 6.0)
    question_term = _clamp01((questions - 1) / 3.0) if questions > 1 else 0.0
    return _clamp01(0.5 * length_term + 0.35 * clause_term + 0.15 * question_term)


def _routing_tier_floor(ctx: RoutingContext) -> int:
    """Task-class floors — mode override is a minimum tier."""
    floor = max(1, int(getattr(ctx, "llm_tier_floor", 1) or 1))
    mode = (getattr(ctx, "mode", None) or "").lower()
    if mode == "root_cause_analysis":
        floor = max(floor, 3)
    elif mode == "crash_investigation":
        floor = max(floor, 3)
    elif mode == "management_view":
        floor = max(floor, 2)
    q = (ctx.query or ctx.prompt or "").lower()
    complex_cues = (
        "differ", "compare", "comparison", "versus", " vs ", "contrast",
        "which dummy", "injury criteria", "injury criterion", "which regulations",
        "which regs", "govern", "mapping", "relate to", "validated by",
        "root cause", "why is", "causal",
    )
    if any(c in q for c in complex_cues):
        floor = max(floor, 2)
    return floor


def _routing_text(ctx: RoutingContext) -> str:
    """User question only — never score tier from the grounded RAG context blob."""
    return (ctx.query or ctx.prompt or "").strip()


def classify(ctx: RoutingContext) -> RouteDecision:
    w = cfg.WEIGHTS
    reasons: list[str] = []
    signals: dict[str, float] = {}

    text = _routing_text(ctx)

    # 1. Query length (not full RAG prompt) ---------------------------------
    tokens = _estimate_tokens(text)
    s_len = _clamp01(tokens / max(1, cfg.PROMPT_TOKENS_SATURATION))
    signals["prompt_length"] = s_len
    if s_len > 0.5:
        reasons.append(f"long query (~{tokens} tokens)")

    # 2. Query complexity ---------------------------------------------------
    s_complex = _query_complexity(text)
    signals["query_complexity"] = s_complex
    if s_complex > 0.5:
        reasons.append("structurally complex / multi-clause query")

    # 3. Grounding confidence (REUSED) — low confidence -> harder task ------
    gc = _clamp01(ctx.grounding_confidence)
    s_ground = _clamp01(1.0 - gc)
    signals["grounding_confidence"] = s_ground
    if gc and gc < 0.5:
        reasons.append(f"low grounding confidence ({gc:.2f}) — needs stronger model")

    # 4. Conversation depth -------------------------------------------------
    s_depth = _clamp01(ctx.conversation_depth / max(1, cfg.CONV_DEPTH_SATURATION))
    signals["conversation_depth"] = s_depth
    if ctx.conversation_depth >= 3:
        reasons.append(f"deep conversation ({ctx.conversation_depth} turns)")

    # 5. Presence of code ---------------------------------------------------
    has_code = ctx.has_code if ctx.has_code is not None else _detect_code(text)
    s_code = 1.0 if has_code else 0.0
    signals["presence_of_code"] = s_code
    if has_code:
        reasons.append("code present")

    # 6. Reasoning tasks ----------------------------------------------------
    hits = _reasoning_hits(text)
    s_reason = _clamp01(len(hits) / 2.0)
    signals["reasoning_tasks"] = s_reason
    if hits:
        reasons.append(f"reasoning cue(s): {', '.join(hits[:3])}")

    # 7. Retrieval confidence (REUSED) — low -> harder ----------------------
    rc = _clamp01(ctx.retrieval_confidence)
    s_retr = _clamp01(1.0 - rc)
    signals["retrieval_confidence"] = s_retr
    if rc and rc < 0.5:
        reasons.append(f"weak retrieval confidence ({rc:.2f})")

    # 8. User feedback history ----------------------------------------------
    s_fb = _clamp01(ctx.feedback_downvote_rate)
    signals["feedback_history"] = s_fb
    if ctx.feedback_downvote_rate > 0.3:
        reasons.append(
            f"recent dissatisfaction ({ctx.feedback_downvote_rate:.0%} 👎) — escalating"
        )

    # 9. Previous model performance (lower success -> escalate) -------------
    #    Use the success rate of Tier 1 as the "can the cheap model cope" proxy.
    t1_success = ctx.model_performance.get(cfg.TIERS[1].model)
    if t1_success is None:
        s_perf = 0.0
    else:
        s_perf = _clamp01(1.0 - float(t1_success))
        if t1_success < 0.6:
            reasons.append(f"Tier 1 historical success low ({t1_success:.0%})")
    signals["model_performance"] = s_perf

    # 10. Estimated cost (tie-break) — cheap query nudges DOWN --------------
    #     Encoded as a small negative pressure proportional to brevity.
    s_cost = _clamp01(1.0 - s_len)  # short/cheap prompts -> reduce score slightly
    signals["estimated_cost"] = s_cost

    weighted = (
        w.prompt_length * s_len
        + w.query_complexity * s_complex
        + w.grounding_confidence * s_ground
        + w.conversation_depth * s_depth
        + w.presence_of_code * s_code
        + w.reasoning_tasks * s_reason
        + w.retrieval_confidence * s_retr
        + w.feedback_history * s_fb
        + w.model_performance * s_perf
        - w.estimated_cost * s_cost  # cost is a damping term
    )
    total_weight = (
        w.prompt_length
        + w.query_complexity
        + w.grounding_confidence
        + w.conversation_depth
        + w.presence_of_code
        + w.reasoning_tasks
        + w.retrieval_confidence
        + w.feedback_history
        + w.model_performance
    )
    # Normalise to 0..10 (cost term can push slightly below the weighted mass).
    score = 10.0 * max(0.0, weighted) / max(1e-9, total_weight)
    score = max(0.0, min(10.0, score))

    # Tier selection is a hybrid of the continuous complexity score and explicit
    # capability floors: some task classes inherently REQUIRE a stronger model
    # regardless of how the weighted signals happen to sum (e.g. code generation
    # and debugging are Tier-3 tasks per the model spec). This keeps routing
    # aligned with the documented tier purposes while the score stays a
    # transparent complexity indicator.
    tier = _tier_for_score(score)
    tier = max(tier, _routing_tier_floor(ctx))
    has_reasoning = len(hits) >= 1
    if has_code and has_reasoning:
        if tier < 3:
            reasons.append("code + reasoning task → escalate to Tier 3")
        tier = max(tier, 3)
    elif has_code or len(hits) >= 2:
        if tier < 2:
            reasons.append("code or multiple reasoning cues → escalate to Tier 2")
        tier = max(tier, 2)

    spec = cfg.TIERS[tier]
    if not reasons:
        reasons.append("simple, well-grounded query → cheapest capable tier")

    return RouteDecision(
        score=score,
        reasons=reasons,
        tier=tier,
        provider=cfg.provider_key_for_tier(tier),
        model=spec.model,
        signals=signals,
    )


def _tier_for_score(score: float) -> int:
    if score < cfg.TIER1_MAX_SCORE:
        return 1
    if score < cfg.TIER2_MAX_SCORE:
        return 2
    return 3


def score_only(ctx: RoutingContext) -> dict:
    """Public helper returning exactly {"score": float, "reasons": [...]}."""
    d = classify(ctx)
    return {"score": round(d.score, 2), "reasons": d.reasons}
