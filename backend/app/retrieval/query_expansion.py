"""
Query expansion + multi-query generation for passive-safety RAG.

Pipeline position:
    User Query -> [Query Expansion] -> [Multi-Query Generation] -> Hybrid Retrieval

- Query Expansion: enrich the query with domain synonyms / abbreviations and
  detect retrieval *intent* (which chunk metadata flags to prefer).
- Multi-Query Generation: produce several query variants (rule-based by default,
  optional LLM paraphrases) so retrieval covers different phrasings.

Rule-based by default -> no LLM dependency / no Groq quota usage.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from loguru import logger

# Domain synonym / expansion groups. Each trigger term pulls in related terms
# that improve BM25 + dense recall on regulation language.
SYNONYM_GROUPS: list[tuple[list[str], list[str]]] = [
    (["strength", "strong", "withstand", "resist"], ["test load", "force", "daN", "traction", "tractive", "load"]),
    (["load", "loading"], ["test load", "daN", "force", "tractive force", "traction device"]),
    (["force"], ["load", "daN", "tractive", "traction"]),
    (["anchorage", "anchor", "mounting"], ["belt anchorage", "anchorage point", "fixing"]),
    (["belt", "seatbelt", "seat-belt"], ["safety-belt", "restraint", "webbing", "harness"]),
    (["dynamic test"], ["sled test", "crash pulse", "deceleration", "impact test"]),
    (["static test"], ["sustained load", "traction device", "applied load"]),
    (["injury"], ["HIC", "chest deflection", "femur force", "thorax", "injury criteria"]),
    (["geometry", "angle", "position"], ["degrees", "location", "effective anchorage"]),
    (["approval", "homologation", "certify"], ["type approval", "approval mark", "compliance"]),
    (["retractor"], ["locking", "emergency locking", "automatic locking"]),
    (["buckle"], ["release", "buckle test", "latch"]),
    (["requirement", "requirements", "spec"], ["shall", "must", "provision", "criteria"]),
]

# Regulation abbreviation expansion.
REG_EXPANSIONS: dict[str, str] = {
    "r14": "UN R14 Regulation No. 14 safety-belt anchorages",
    "r16": "UN R16 Regulation No. 16 safety-belts restraint systems",
    "r17": "UN R17 Regulation No. 17 seat strength",
    "r94": "UN R94 frontal collision",
    "r95": "UN R95 lateral collision",
    "ncap": "Euro NCAP crash test rating",
}

# Intent -> chunk metadata feature flags to prefer during metadata filtering.
INTENT_FLAGS: list[tuple[list[str], list[str]]] = [
    (["load", "strength", "force", "dan", "tractive", "withstand"], ["has_loads", "has_test_procedure"]),
    (["test", "dynamic", "static", "procedure", "sled", "pulse"], ["has_test_procedure"]),
    (["injury", "hic", "chest", "femur", "thorax", "neck"], ["has_injury_criteria"]),
    (["angle", "geometry", "degree", "position", "location"], ["has_angles"]),
    (["anchorage", "belt", "retractor", "buckle", "restraint", "webbing"], ["has_belt_system"]),
    (["vehicle", "category", "m1", "n1", "m3", "n3"], ["has_vehicle_classes"]),
    (["distance", "dimension", "mm", "length", "width"], ["has_distances"]),
    (["shall", "must", "requirement", "comply", "mandatory"], ["has_requirements"]),
]

STOPWORDS = {
    "what", "which", "when", "where", "how", "the", "for", "are", "and", "under",
    "with", "that", "this", "from", "into", "does", "your", "about", "explain",
    "describe", "list", "give", "tell", "purpose", "requirements", "requirement",
}


@dataclass
class ExpandedQuery:
    original: str
    expanded: str
    intent_flags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


def is_comparison_query(query: str) -> bool:
    """True when the user is comparing or contrasting regulations/tests."""
    q = query.lower()
    return bool(re.search(
        r"\b(?:differ(?:s|ence)?|compare|comparison|contrast|versus|\bvs\.?\b|"
        r"how\s+does|how\s+do|distinguish|between)\b",
        q,
    ))


def _tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9.]", " ", text.lower()).split()


def detect_intent_flags(query: str) -> list[str]:
    """Return chunk metadata flags that match the query intent."""
    q = query.lower()
    flags: list[str] = []
    for triggers, group_flags in INTENT_FLAGS:
        if any(t in q for t in triggers):
            for f in group_flags:
                if f not in flags:
                    flags.append(f)
    return flags


def _keywords(query: str) -> list[str]:
    toks = _tokens(query)
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def expand_query(query: str) -> ExpandedQuery:
    """Enrich the query with domain synonyms + regulation expansions."""
    q_low = query.lower()
    extra: list[str] = []

    for triggers, additions in SYNONYM_GROUPS:
        if any(t in q_low for t in triggers):
            extra.extend(additions)

    for abbr, expansion in REG_EXPANSIONS.items():
        if re.search(rf"\br?{abbr}\b", q_low) or abbr in q_low.replace(" ", ""):
            extra.append(expansion)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    extra_unique = [x for x in extra if not (x in seen or seen.add(x))]

    expanded = query if not extra_unique else f"{query} {' '.join(extra_unique)}"
    return ExpandedQuery(
        original=query,
        expanded=expanded,
        intent_flags=detect_intent_flags(query),
        keywords=_keywords(query),
    )


def _llm_multi_queries(query: str, n: int) -> list[str]:
    """Optional LLM paraphrases (only if ENABLE_LLM_MULTI_QUERY=true)."""
    try:
        from backend.app.llm.groq_client import GroqLLM

        prompt = (
            "Rewrite the following passive-safety regulation question into "
            f"{n} short alternative search queries (one per line, no numbering). "
            f"Keep technical terms.\n\nQUESTION: {query}"
        )
        out = GroqLLM().generate(prompt)["answer"]
        variants = [
            re.sub(r"^[\d\.\)\-\s]+", "", line).strip()
            for line in out.splitlines()
            if line.strip()
        ]
        return [v for v in variants if v][:n]
    except Exception as exc:
        logger.warning(f"LLM multi-query skipped: {exc}")
        return []


def generate_multi_queries(query: str, expanded: ExpandedQuery | None = None) -> list[str]:
    """
    Produce several query variants for multi-query retrieval.

    Default = rule-based (original + expanded + keyword-focused + intent-focused).
    Set ENABLE_LLM_MULTI_QUERY=true to add Groq paraphrases.
    """
    exp = expanded or expand_query(query)
    n = int(os.getenv("MULTI_QUERY_COUNT", "3"))

    variants: list[str] = [query.strip()]

    if exp.expanded != query:
        variants.append(exp.expanded)

    # Keyword-focused variant (drops question words).
    if exp.keywords:
        kw = " ".join(exp.keywords)
        if kw and kw not in variants:
            variants.append(kw)

    # Intent-focused variant: keywords + intent terms.
    intent_terms = {
        "has_loads": "test load force daN",
        "has_test_procedure": "test procedure",
        "has_injury_criteria": "injury criteria",
        "has_belt_system": "belt anchorage restraint",
        "has_angles": "angle geometry",
    }
    intent_extra = " ".join(intent_terms[f] for f in exp.intent_flags if f in intent_terms)
    if intent_extra and exp.keywords:
        variants.append(f"{' '.join(exp.keywords)} {intent_extra}")

    if os.getenv("ENABLE_LLM_MULTI_QUERY", "false").lower() == "true":
        variants.extend(_llm_multi_queries(query, n))

    # Deduplicate, cap to n+1 (original always kept).
    seen: set[str] = set()
    unique = [v for v in variants if v and not (v.lower() in seen or seen.add(v.lower()))]
    return unique[: max(2, n + 1)]
