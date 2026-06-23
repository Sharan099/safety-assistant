"""Soft-boost retrieval when query names a vehicle category or anchorage test type."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_QUERY_RE = {
    "M1_N1": re.compile(r"\bM1\b|\bN1\b|M1/N1|M1\s+and\s+N1", re.I),
    "M3_N3": re.compile(r"\bM3\b|\bN3\b|M3/N3|M3\s+and\s+N3", re.I),
    "M2_N2": re.compile(r"\bM2\b|\bN2\b|M2/N2", re.I),
    "NOT_M1_N1": re.compile(r"other\s+than\s+M1|M2|M3|N2|N3|heavy\s+vehicle|bus|truck", re.I),
}

_ANCHORAGE_QUERY_RE = {
    "upper_torso_strap_anchorage": re.compile(
        r"upper\s+torso|upper\s+belt\s+anchorage|torso\s+strap", re.I
    ),
    "lower_anchorage_traction": re.compile(
        r"lower\s+anchorage|lower\s+belt|tractive\s+force.*lower", re.I
    ),
    "lap_belt_lower": re.compile(r"lap\s+belt", re.I),
    "special_type_belt_upper": re.compile(r"special[- ]type\s+belt", re.I),
}


def detect_query_categories(query: str) -> set[str]:
    q = query or ""
    found: set[str] = set()
    for token, pat in _CATEGORY_QUERY_RE.items():
        if pat.search(q):
            found.add(token)
    if "M3_N3" in found or "M2_N2" in found:
        found.add("NOT_M1_N1")
    return found


def detect_query_anchorage_type(query: str) -> str | None:
    q = query or ""
    for test_type, pat in _ANCHORAGE_QUERY_RE.items():
        if pat.search(q):
            return test_type
    if re.search(r"\banchorage\b", q, re.I) and not re.search(r"upper|lower|lap", q, re.I):
        return None
    return None


def applicability_soft_boost(chunk: dict[str, Any], query: str) -> float:
    """Rank multiplier when chunk applicability matches explicit query signals."""
    boost = 1.0
    q_cats = detect_query_categories(query)
    chunk_cats = chunk.get("applies_to_category") or []
    if isinstance(chunk_cats, str):
        chunk_cats = [chunk_cats]

    if q_cats and chunk_cats:
        overlap = q_cats.intersection(chunk_cats)
        if overlap:
            boost *= 1.25
        elif "M3_N3" in q_cats and "NOT_M1_N1" in chunk_cats:
            boost *= 1.12
        elif "M1_N1" in q_cats and "M1_N1" not in chunk_cats and "NOT_M1_N1" in chunk_cats:
            boost *= 0.75

    q_test = detect_query_anchorage_type(query)
    chunk_test = chunk.get("anchorage_test_type")
    if q_test and chunk_test:
        if q_test == chunk_test:
            boost *= 1.2
        elif q_test in ("lower_anchorage_traction",) and chunk_test in (
            "lower_anchorage_no_retractor",
            "special_type_belt_lower",
        ):
            boost *= 1.1

    if chunk.get("has_duration_requirement") and re.search(
        r"duration|hold\s*time|0\.2\s*s|second", query, re.I
    ):
        boost *= 1.15

    return boost
