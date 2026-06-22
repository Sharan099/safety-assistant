"""
Query intent detection — test_type, region, doc_type, value_type intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.document_registry import get_document_meta
from backend.app.retrieval.clause_topic import (
    chunk_passes_topic_filter,
    detect_allowed_clause_topics,
)

DOC_TYPE_LEGAL = "legal"
DOC_TYPE_RATING = "rating"
DOC_TYPE_REFERENCE = "reference"


@dataclass
class QueryIntent:
    test_type: str | None = None
    region: str | None = None
    doc_type_intent: str | None = None  # legal | rating | reference | any
    value_type_intent: str | None = None  # legal_limit | rating_threshold | target | any
    regulation_codes: list[str] = field(default_factory=list)
    exclude_doc_types: list[str] = field(default_factory=list)
    exclude_value_types: list[str] = field(default_factory=list)
    allowed_clause_topics: frozenset[str] | None = None
    requirement_cluster: str | None = None
    raw_query: str = ""


REG_MAP = {
    "un r14": "UN_R14", "r14": "UN_R14",
    "un r16": "UN_R16", "r16": "UN_R16",
    "un r17": "UN_R17", "r17": "UN_R17",
    "un r94": "UN_R94", "r94": "UN_R94",
    "un r95": "UN_R95", "r95": "UN_R95",
    "un r135": "UN_R135", "r135": "UN_R135",
    "un r137": "UN_R137", "r137": "UN_R137",
    "fmvss": "FMVSS", "fmvss 208": "FMVSS",
    "euro ncap": "EURO_NCAP", "euroncap": "EURO_NCAP", "ncap": "EURO_NCAP",
}

_TEST_PATTERNS = {
    "frontal": ("frontal", "odb", "head-on", "full-width frontal", "r94", "r137", "fmvss 208"),
    "side": ("side impact", "lateral", "r95", "worldsid", "es-2"),
    "pole_side": ("pole", "r135", "pole side"),
    "rear": ("rear impact", "rear-end", "whiplash"),
    "pedestrian": ("pedestrian", "vru", "walker"),
    "belt": ("belt", "anchorage", "restraint", "r14", "r16"),
    "seat": ("seat", "head restraint", "r17"),
}


def detect_query_intent(query: str) -> QueryIntent:
    q = query.lower()
    intent = QueryIntent(raw_query=query)

    # Regulation codes
    for key, code in REG_MAP.items():
        if key in q and code not in intent.regulation_codes:
            intent.regulation_codes.append(code)

    # Test type
    for test_type, keywords in _TEST_PATTERNS.items():
        if any(kw in q for kw in keywords):
            intent.test_type = test_type
            break

    # Region — if query spans EU + US regulations, do not hard-filter by region.
    eu_hit = any(k in q for k in ("europe", "eu ", " unece", "ece ", "un r"))
    us_hit = any(k in q for k in ("us ", "usa", "nhtsa", "fmvss", "united states"))
    named_eu = any(c.startswith("UN_") for c in intent.regulation_codes)
    named_us = "FMVSS" in intent.regulation_codes
    if (eu_hit and us_hit) or (named_eu and named_us):
        intent.region = None
    elif eu_hit:
        intent.region = "EU"
    elif us_hit:
        intent.region = "US"

    # Doc type intent
    if any(k in q for k in (
        "legal", "regulation", "requirement", "shall", "binding",
        "approval", "limit", "maximum", "minimum", "not exceed",
    )):
        intent.doc_type_intent = DOC_TYPE_LEGAL
        intent.value_type_intent = "legal_limit"
        intent.exclude_doc_types = [DOC_TYPE_REFERENCE]
        intent.exclude_value_types = ["rating_threshold"]
    elif any(k in q for k in (
        "euro ncap", "ncap score", "rating", "points", "star",
        "assessment protocol", "scoring",
    )):
        intent.doc_type_intent = DOC_TYPE_RATING
        intent.value_type_intent = "rating_threshold"
    elif any(k in q for k in ("target", "internal", "design objective", "program")):
        intent.doc_type_intent = DOC_TYPE_REFERENCE
        intent.value_type_intent = "target"

    # Explicit legal limit phrasing
    if "legal limit" in q or "legal requirement" in q or "official requirement" in q:
        intent.doc_type_intent = DOC_TYPE_LEGAL
        intent.value_type_intent = "legal_limit"
        intent.exclude_doc_types = [DOC_TYPE_REFERENCE, DOC_TYPE_RATING]
        intent.exclude_value_types = ["rating_threshold", "target"]

    intent.allowed_clause_topics = detect_allowed_clause_topics(query)
    intent.requirement_cluster = _detect_requirement_cluster(query)

    return intent


def _detect_requirement_cluster(query: str) -> str | None:
    from backend.app.core.document_registry import match_requirement_cluster
    return match_requirement_cluster(query)


def chunk_passes_intent_filter(chunk: dict[str, Any], intent: QueryIntent) -> bool:
    """Hard pre-filter: return False if chunk must be excluded."""
    if intent.exclude_doc_types:
        if chunk.get("doc_type") in intent.exclude_doc_types:
            return False
    if intent.exclude_value_types and chunk.get("value_type"):
        if chunk.get("value_type") in intent.exclude_value_types:
            return False

    if intent.test_type and chunk.get("test_type"):
        ct = chunk.get("test_type", "general")
        if ct != "general" and ct != intent.test_type:
            # belt/seat are orthogonal to impact direction — allow if query is belt/seat
            if intent.test_type in ("belt", "seat"):
                if ct not in (intent.test_type, "general"):
                    return False
            elif intent.test_type in ("frontal", "side", "pole_side", "rear", "pedestrian"):
                if ct in ("frontal", "side", "pole_side", "rear", "pedestrian") and ct != intent.test_type:
                    return False

    if intent.region and chunk.get("region"):
        cr = chunk.get("region")
        if cr not in (intent.region, "global"):
            return False

    if intent.doc_type_intent and chunk.get("doc_type"):
        if intent.doc_type_intent == DOC_TYPE_LEGAL and chunk.get("doc_type") == DOC_TYPE_REFERENCE:
            if intent.value_type_intent == "legal_limit":
                return False

    if not chunk_passes_topic_filter(chunk, intent.allowed_clause_topics):
        return False

    return True
