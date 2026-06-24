"""
Query decomposition for comparative and compound regulation questions.

Splits multi-regulation questions into per-regulation sub-queries, retrieves each
independently, then merges before reranking / generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.app.core.document_registry import detect_regulations_in_query
from backend.app.retrieval.query_expansion import is_comparison_query

_COMPOUND_RE = re.compile(
    r"\b(?:and|both|as well as|plus|&)\b",
    re.I,
)
_MULTI_IMPACT_RE = re.compile(
    r"\b(frontal|side|lateral|pole|rear|pedestrian)\b.*\b(?:and|&|,)\b.*\b(frontal|side|lateral|pole|rear|pedestrian)\b",
    re.I,
)


@dataclass
class DecomposedQuery:
    original: str
    sub_queries: list[str] = field(default_factory=list)
    target_regulations: list[str] = field(default_factory=list)
    is_comparative: bool = False
    is_compound: bool = False


def _impact_phrases(query: str) -> list[str]:
    q = query.lower()
    found: list[str] = []
    for label, patterns in (
        ("frontal impact", ("frontal", "head-on", "odb")),
        ("side impact", ("side impact", "lateral collision")),
        ("pole side impact", ("pole side", "pole impact", "psi")),
        ("rear impact", ("rear impact", "rear-end")),
        ("pedestrian protection", ("pedestrian", "vru")),
    ):
        if any(p in q for p in patterns):
            found.append(label)
    return found


def decompose_query(query: str) -> DecomposedQuery:
    """Return sub-queries when the question names multiple regs or impact modes."""
    regs = detect_regulations_in_query(query)
    comparative = is_comparison_query(query)
    compound = bool(
        len(regs) >= 2
        or (_COMPOUND_RE.search(query) and len(regs) >= 1)
        or _MULTI_IMPACT_RE.search(query)
    )

    result = DecomposedQuery(
        original=query,
        is_comparative=comparative,
        is_compound=compound,
        target_regulations=list(regs),
    )

    if not compound and not comparative:
        result.sub_queries = [query]
        return result

    sub: list[str] = []

    if len(regs) >= 2:
        for code in regs:
            display = code.replace("_", " ")
            sub.append(f"{display}: {query}")
    elif comparative and "euro ncap" in query.lower() and any(r.startswith("UN_R") for r in regs):
        sub.append(f"Euro NCAP protocol: {query}")
        for code in regs:
            if code.startswith("UN_R"):
                sub.append(f"{code.replace('_', ' ')} legal requirements: {query}")
    elif impacts := _impact_phrases(query):
        for imp in impacts:
            sub.append(f"{imp} — {query}")
    else:
        sub.append(query)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result.sub_queries = [s for s in sub if not (s in seen or seen.add(s))]
    if not result.sub_queries:
        result.sub_queries = [query]
    return result
