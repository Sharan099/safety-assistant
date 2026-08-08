"""Per-criterion retrieval for multi-measurement / multi-criteria queries.

When a query names 2+ distinct criteria (HPC, ThCC, fuel leakage, …), run a
separate hybrid search per criterion (default top-3 each) and merge/dedupe so
later criteria are not crowded out of a shared top-k budget.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_PER_CRITERION_TOP_K = 3

# Criterion key → (match regex, focused retrieval query string).
# Longer / more specific patterns first where they overlap.
_CRITERION_SPECS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "pspf",
        re.compile(r"(?i)\bpubic\s+symphysis(?:\s+peak\s+force)?\b|\bPSPF\b"),
        "Pubic Symphysis Peak Force (PSPF) pelvis performance criterion injury criteria 6 kN",
    ),
    (
        "rdc",
        re.compile(r"(?i)\brib\s+deflection(?:\s+criterion)?\b|\bRDC\b"),
        "Rib Deflection Criterion (RDC) injury criteria performance criteria 42 mm",
    ),
    (
        "thcc",
        re.compile(
            r"(?i)\bthorax\s+compression(?:\s+criterion)?\b|\bThCC\b|\bTHCC\b|"
            r"\bchest\s+compression\b"
        ),
        # Include RDC: R95 side-impact chest limits use Rib Deflection, not ThCC.
        "Thorax Compression Criterion (ThCC) chest compression "
        "Rib Deflection Criterion (RDC) injury criteria 42 mm",
    ),
    (
        "hpc",
        re.compile(
            r"(?i)\bhead\s+performance(?:\s+criterion)?\b|\bHPC(?:36)?\b|"
            r"\bHIC(?:15|36)?\b|\bhead\s+injury\b"
        ),
        "Head Performance Criterion (HPC) injury criteria shall not exceed 1000",
    ),
    (
        "vc",
        re.compile(
            r"(?i)\bviscous\s+criterion\b|\bsoft\s+tissue\s+criterion\b|"
            r"\bVC\b|\bV\s*\*\s*C\b"
        ),
        "Viscous Criterion (VC) Soft Tissue Criterion injury criteria 1.0 m/s",
    ),
    (
        "tti",
        re.compile(r"(?i)\bthoracic\s+trauma\b|\bTTI\b"),
        "Thoracic Trauma Index (TTI) injury criteria performance criteria",
    ),
    (
        "fuel_leakage",
        re.compile(
            r"(?i)\bfuel\s*[- ]?\s*leak(?:age)?(?:\s+rate)?\b|"
            r"\bleakage\s+rate\b|\bfuel[- ]?feed\b"
        ),
        "fuel-feed installation leakage rate shall not exceed 30 g/min continuous leakage",
    ),
    (
        "electrolyte_leakage",
        re.compile(
            r"(?i)\belectrolyte\b|\bspillage\b|"
            r"\belectrolyte\s+leak(?:age)?\b"
        ),
        "electrolyte leakage REESS passenger compartment shall be no liquid "
        "electrolyte leakage into the passenger compartment spillage",
    ),
]

# Keywords that should appear in retrieved text for each criterion key
# (used by golden / regression checks).
CRITERION_CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "hpc": ["hpc", "head performance"],
    "thcc": [
        "thcc",
        "thorax compression",
        "chest compression",
        "rib deflection",
        "rdc",
        "thorax performance",
    ],
    "rdc": ["rdc", "rib deflection"],
    "vc": ["viscous", "soft tissue", "vc"],
    "pspf": ["pspf", "pubic"],
    "tti": ["tti", "thoracic trauma"],
    "fuel_leakage": ["fuel", "leakage", "g/min", "30 g"],
    "electrolyte_leakage": [
        "electrolyte",
        "passenger compartment",
        "leakage",
        "spillage",
    ],
}


@dataclass(frozen=True)
class NamedCriterion:
    key: str
    matched_text: str
    retrieval_query: str
    context_keywords: list[str] = field(default_factory=list)


def list_named_criteria(question: str) -> list[NamedCriterion]:
    """Return distinct criteria named in the question (stable cue order)."""
    q = question or ""
    if not q.strip():
        return []
    found: list[NamedCriterion] = []
    seen: set[str] = set()
    for key, pat, retrieval_q in _CRITERION_SPECS:
        m = pat.search(q)
        if not m:
            continue
        if key in seen:
            continue
        seen.add(key)
        found.append(
            NamedCriterion(
                key=key,
                matched_text=m.group(0),
                retrieval_query=retrieval_q,
                context_keywords=list(CRITERION_CONTEXT_KEYWORDS.get(key, [])),
            )
        )
    return found


def is_multi_criterion_query(question: str) -> bool:
    """True when the question names two or more distinct criteria."""
    return len(list_named_criteria(question)) >= 2


def per_criterion_top_k() -> int:
    try:
        return max(1, int(os.getenv("MULTI_CRITERION_TOP_K", str(DEFAULT_PER_CRITERION_TOP_K))))
    except ValueError:
        return DEFAULT_PER_CRITERION_TOP_K


def merge_per_criterion_chunks(
    per_criterion: Sequence[Sequence[T]],
    *,
    per_k: int | None = None,
) -> list[T]:
    """Take up to ``per_k`` from each list, dedupe by chunk_id, preserve order.

    Round-robin across criteria so early criteria cannot monopolize the merge
    when the same mega-chunk appears in every list.
    """
    k = per_k if per_k is not None else per_criterion_top_k()
    trimmed: list[list[T]] = []
    for ranked in per_criterion:
        trimmed.append(list(ranked)[:k])

    out: list[T] = []
    seen: set[str] = set()
    max_len = max((len(t) for t in trimmed), default=0)
    for rank_i in range(max_len):
        for ranked in trimmed:
            if rank_i >= len(ranked):
                continue
            chunk = ranked[rank_i]
            cid = str(getattr(chunk, "chunk_id", None) or id(chunk))
            if cid in seen:
                continue
            seen.add(cid)
            out.append(chunk)
    return out


def criterion_covered_by_chunks(
    criterion: NamedCriterion,
    chunks: Sequence[object],
) -> bool:
    """True if any chunk text/title mentions a keyword for this criterion."""
    keys = [k.lower() for k in criterion.context_keywords] or [
        criterion.matched_text.lower()
    ]
    for c in chunks:
        blob = " ".join(
            str(getattr(c, attr, "") or "")
            for attr in ("text", "enriched_text", "section_title", "section_number")
        ).lower()
        if any(k in blob for k in keys if k):
            return True
    return False


def uncovered_criteria(
    question: str,
    chunks: Sequence[object],
) -> list[NamedCriterion]:
    """Criteria named in the question that do not appear in retrieved context."""
    return [
        c
        for c in list_named_criteria(question)
        if not criterion_covered_by_chunks(c, chunks)
    ]
