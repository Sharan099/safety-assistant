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


@dataclass
class CrossDocHop:
    hop_id: int
    label: str
    query: str
    target_doc_types: list[str] = field(default_factory=list)
    authority_role: str = "advisory"


@dataclass
class CrossDocumentDecomposition:
    original: str
    hops: list[CrossDocHop] = field(default_factory=list)
    is_cross_document: bool = False


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


def decompose_cross_document(query: str) -> CrossDocumentDecomposition:
    """
    Split cross-document compliance questions into hops targeting doc types.
    e.g. regulation limit (legal) + crash report measurement (test_report).
    """
    q = query.lower()
    result = CrossDocumentDecomposition(original=query)

    measured = bool(re.search(r"\b(measured|crash report|test report|our|actual|observed)\b", q))
    requirement = bool(re.search(r"\b(limit|requirement|regulation|standard|shall|must)\b", q))
    compare = bool(re.search(r"\b(meet|comply|compare|versus|vs|against|exceed)\b", q))

    if measured and requirement and compare:
        result.is_cross_document = True
        result.hops = [
            CrossDocHop(
                hop_id=1,
                label="Binding requirement",
                query=f"Legal/regulatory requirement and limit: {query}",
                target_doc_types=["legal"],
                authority_role="binding",
            ),
            CrossDocHop(
                hop_id=2,
                label="Measured / observed value",
                query=f"Measured test result or observed value: {query}",
                target_doc_types=["test_report", "sim_report", "internal"],
                authority_role="measured",
            ),
        ]
        return result

    # Fallback: regulation vs rating comparison across uploads
    regs = detect_regulations_in_query(query)
    if len(regs) >= 2 and is_comparison_query(query):
        result.is_cross_document = True
        for i, code in enumerate(regs[:3], 1):
            result.hops.append(
                CrossDocHop(
                    hop_id=i,
                    label=code.replace("_", " "),
                    query=f"{code.replace('_', ' ')}: {query}",
                    target_doc_types=[],
                    authority_role="binding" if code.startswith("UN_R") else "advisory",
                )
            )
        return result

    result.hops = [
        CrossDocHop(
            hop_id=1,
            label="General retrieval",
            query=query,
            target_doc_types=[],
            authority_role="advisory",
        )
    ]
    return result
