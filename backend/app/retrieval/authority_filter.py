"""Authority-tier hard filters — prevent binding/advisory source confusion."""

from __future__ import annotations

import re
from typing import Any

from backend.app.core.authority_tier import (
    ENGINEERING_REF,
    HISTORICAL_DATA,
    LEGAL_BINDING,
    OEM_INTERNAL,
    RATING_PROTOCOL,
    SYNTHETIC,
    chunk_authority_tier,
    is_binding_tier,
)
from backend.app.retrieval.query_intent import QueryIntent

_COMPLIANCE_QUERY_RE = re.compile(
    r"\b(compliant|compliance|required|must comply|is it required|legally required|"
    r"binding|shall|mandatory|approval requirement|legal requirement|"
    r"does .+ require|is .+ required)\b",
    re.I,
)


def is_compliance_determination_query(query: str) -> bool:
    return bool(_COMPLIANCE_QUERY_RE.search(query or ""))


def binding_tier_filter(intent: QueryIntent | None = None) -> frozenset[str]:
    """Tiers allowed when determining legal compliance / binding requirements."""
    return frozenset({LEGAL_BINDING})


def supporting_tiers_for_compliance() -> frozenset[str]:
    """Non-binding tiers retrievable as separated supporting context only."""
    return frozenset({
        RATING_PROTOCOL,
        ENGINEERING_REF,
        OEM_INTERNAL,
        HISTORICAL_DATA,
        SYNTHETIC,
    })


def chunk_passes_authority_filter(
    chunk: dict[str, Any],
    *,
    compliance_query: bool,
    binding_only: bool = False,
    allowed_tiers: frozenset[str] | None = None,
) -> bool:
    tier = chunk_authority_tier(chunk)
    if allowed_tiers is not None and tier not in allowed_tiers:
        return False
    if binding_only and not is_binding_tier(tier):
        return False
    if compliance_query and binding_only:
        return is_binding_tier(tier)
    return True


def split_docs_by_authority_tier(
    documents: list[dict],
    citations: list[dict],
) -> dict[str, list[tuple[dict, dict]]]:
    """Group (doc, citation) pairs by authority_tier for prompt injection."""
    buckets: dict[str, list[tuple[dict, dict]]] = {
        LEGAL_BINDING: [],
        RATING_PROTOCOL: [],
        ENGINEERING_REF: [],
        OEM_INTERNAL: [],
        HISTORICAL_DATA: [],
        SYNTHETIC: [],
    }
    for d, c in zip(documents, citations):
        tier = chunk_authority_tier(d)
        if tier not in buckets:
            tier = ENGINEERING_REF
        buckets[tier].append((d, c))
    return {k: v for k, v in buckets.items() if v}


_TIER_SECTION_HEADERS: dict[str, str] = {
    LEGAL_BINDING: "=== BINDING LEGAL REQUIREMENTS (legal_binding — must comply) ===",
    RATING_PROTOCOL: "=== RATING PROTOCOLS (rating_protocol — NOT legally binding) ===",
    ENGINEERING_REF: "=== ENGINEERING REFERENCES (engineering_ref — advisory only) ===",
    OEM_INTERNAL: "=== OEM INTERNAL STANDARDS (oem_internal — proprietary advisory) ===",
    HISTORICAL_DATA: "=== HISTORICAL EVIDENCE (historical_data — past cases, not requirements) ===",
    SYNTHETIC: "=== SYNTHETIC DATA (synthetic — placeholder, not regulatory) ===",
}
