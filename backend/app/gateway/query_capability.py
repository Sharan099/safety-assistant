"""Detect queries unsafe for the fast (8B) tier and escalate routing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.gateway import config as cfg
from backend.app.gateway.fallback_safeguards import is_fast_groq_model
from backend.app.gateway.types import RouteDecision

_UNIT_RE = re.compile(
    r"\b(convert|conversion|kN|daN|newton|pound|lb|kg|mm|cm|m\b|mph|km/h|"
    r"percent|%|tolerance|±|plus or minus|arithmetic|calculate|compute)\b",
    re.I,
)
_ARITH_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(?:daN|kN|N|mm|cm|m)\b.*\b(?:to|in|as|into)\b",
    re.I,
)
_DEFINE_DISTINGUISH_RE = re.compile(
    r"\b(differ|difference|distinguish|compare|versus|vs\.?|contrast|"
    r"definition|defined as|what is the difference|how does .+ differ from)\b",
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
    text = f"{query}\n{prompt}".lower()
    signals: list[str] = []
    arith = bool(_UNIT_RE.search(text) or _ARITH_RE.search(text))
    if arith:
        signals.append("unit_conversion_or_arithmetic")
    def_dist = bool(_DEFINE_DISTINGUISH_RE.search(text))
    term_hits = len(set(_TERM_PAIRS_RE.findall(text)))
    if term_hits >= 2 and def_dist:
        signals.append("multi_term_definition_distinction")
    elif def_dist and "definition" in text:
        signals.append("definition_distinction")
    return QueryCapability(
        requires_arithmetic=arith,
        requires_definition_distinction=def_dist or term_hits >= 2,
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
