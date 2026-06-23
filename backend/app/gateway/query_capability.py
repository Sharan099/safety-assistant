"""Detect queries unsafe for the fast (8B) tier and escalate routing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.gateway import config as cfg
from backend.app.gateway.fallback_safeguards import is_fast_groq_model
from backend.app.gateway.types import RouteDecision

# Explicit conversion / calculation intent — NOT bare regulatory units (daN, mm, etc.).
_ARITH_INTENT_RE = re.compile(
    r"\b(convert|conversion|calculate|computed?|compute|arithmetic|"
    r"how many|express .+ in|what is .+ in (?:kN|daN|newtons?))\b",
    re.I,
)
_ARITH_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(?:daN|kN|N|mm|cm|m)\b.*\b(?:to|in|as|into)\b",
    re.I,
)
_DEFINE_DISTINGUISH_RE = re.compile(
    r"\b(differ|difference|distinguish|compare|versus|vs\.?|contrast|"
    r"defined as|what is the difference|how does .+ differ from)\b",
    re.I,
)
_TERM_PAIRS_RE = re.compile(
    r"\b(anchorage|belt anchorage|effective anchorage|upper anchorage|"
    r"lower anchorage|retractor|belt|safety.?belt)\b",
    re.I,
)


@dataclass(frozen=True)
class QueryCapability:
    requires_arithmetic: bool = False
    requires_definition_distinction: bool = False
    matched_signals: tuple[str, ...] = ()

    @property
    def unsafe_for_fast_tier(self) -> bool:
        return self.requires_arithmetic or self.requires_definition_distinction


def assess_query_capability(query: str, prompt: str = "") -> QueryCapability:
    """Assess capability from the user question only.

    The optional ``prompt`` arg is ignored for routing. RAG workflows pass a long
    grounded prompt full of daN values, anchorage terms, and "compare" language
    from retrieved regulations — scanning it caused every production query to
    escalate to Tier 3 (Sonnet).
    """
    del prompt  # intentionally query-only; kept for call-site compatibility
    text = (query or "").lower()
    if not text:
        return QueryCapability()

    signals: list[str] = []
    arith = bool(_ARITH_INTENT_RE.search(text) or _ARITH_RE.search(text))
    if arith:
        signals.append("unit_conversion_or_arithmetic")

    def_dist = bool(_DEFINE_DISTINGUISH_RE.search(text))
    term_hits = len(set(_TERM_PAIRS_RE.findall(text)))
    needs_def = def_dist and term_hits >= 2
    if needs_def:
        signals.append("multi_term_definition_distinction")
    elif def_dist and "definition" in text:
        signals.append("definition_distinction")
        needs_def = True

    return QueryCapability(
        requires_arithmetic=arith,
        requires_definition_distinction=needs_def,
        matched_signals=tuple(signals),
    )


def apply_capability_escalation(
    decision: RouteDecision,
    capability: QueryCapability,
) -> tuple[RouteDecision, list[str]]:
    """Bump tier floor when 8B cannot safely answer."""
    if not capability.unsafe_for_fast_tier:
        return decision, []

    reasons = list(decision.reasons)
    tier = decision.tier
    escalations: list[str] = []

    if capability.requires_arithmetic:
        tier = max(tier, 2)
        escalations.append("arithmetic/unit conversion → minimum tier 2 (70B+)")
    if capability.requires_definition_distinction:
        tier = max(tier, 2)
        escalations.append("definition distinction → minimum tier 2 (70B+)")

    if is_fast_groq_model(decision.model):
        tier = max(tier, 2)
        escalations.append("blocked fast-tier model for capability class")

    if tier == decision.tier:
        return decision, escalations

    spec = cfg.TIERS[tier]
    reasons.extend(escalations)
    return RouteDecision(
        score=decision.score,
        reasons=reasons,
        tier=tier,
        provider=cfg.provider_key_for_tier(tier),
        model=spec.model,
        signals=decision.signals,
    ), escalations


def fast_tier_blocked_for_capability(capability: QueryCapability) -> bool:
    return capability.unsafe_for_fast_tier
