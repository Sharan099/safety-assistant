"""Query analysis and rewriting for definition- and comparison-style questions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_DEFINITION_PATTERNS = [
    re.compile(r"^\s*what\s+is\s+(?:a|an|the\s+)?(.+?)\s*\??\s*$", re.I),
    re.compile(r"^\s*what\s+does\s+(.+?)\s+mean\s*\??\s*$", re.I),
    re.compile(r"^\s*define\s+(?:the\s+)?(.+?)\s*\??\s*$", re.I),
    re.compile(r"^\s*definition\s+of\s+(?:the\s+)?(.+?)\s*\??\s*$", re.I),
    re.compile(r"^\s*how\s+is\s+(.+?)\s+defined\s*\??\s*$", re.I),
]

_COMPARISON_RE = re.compile(
    r"(?:"
    r"compare|comparison|difference(?:s)?\s+between|versus|\bvs\.?\b|"
    r"contrast|how\s+do(?:es)?\s+.+\s+differ"
    r")",
    re.I,
)

_TERM_EXTRACT_RE = re.compile(
    r"\b(?:UN[\s_-]?)?(R\d{2,3})\b",
    re.I,
)


@dataclass
class QueryAnalysis:
    original: str
    is_definition: bool = False
    is_comparison: bool = False
    defined_term: str | None = None
    regulation_codes: list[str] = field(default_factory=list)
    retrieval_query: str = ""
    sparse_terms: list[str] = field(default_factory=list)


def _clean_term(term: str) -> str:
    term = re.sub(r"\s+", " ", term.strip().rstrip("?."))
    term = re.sub(r"^(?:the|a|an)\s+", "", term, flags=re.I)
    return term.strip()


def analyze_query(query: str) -> QueryAnalysis:
    q = query.strip()
    analysis = QueryAnalysis(original=q, retrieval_query=q)

    for pattern in _DEFINITION_PATTERNS:
        match = pattern.match(q)
        if match:
            analysis.is_definition = True
            analysis.defined_term = _clean_term(match.group(1))
            break

    if _COMPARISON_RE.search(q):
        analysis.is_comparison = True

    # Use shared detector — must match regulation_detect (R94 + "Regulation No. 94").
    from core.regulation_detect import regulation_codes as detect_regulation_codes

    analysis.regulation_codes = detect_regulation_codes(q)

    sparse_terms: list[str] = []
    if analysis.is_definition and analysis.defined_term:
        term = analysis.defined_term
        sparse_terms.extend(
            [
                f"definition of {term}",
                f"{term} means",
                f"{term} shall mean",
                "definitions",
                "2. Definitions",
            ]
        )
        analysis.retrieval_query = (
            f"{q} definition of {term} means shall mean 2. Definitions {term}"
        )
    elif analysis.is_comparison:
        sparse_terms.extend(["requirements", "shall", "comparison"])
        if analysis.regulation_codes:
            sparse_terms.extend(analysis.regulation_codes)
        analysis.retrieval_query = f"{q} {' '.join(analysis.regulation_codes)}"
    else:
        analysis.retrieval_query = q

    analysis.sparse_terms = sparse_terms
    return analysis


def sparse_query_text(analysis: QueryAnalysis) -> str:
    """FTS-oriented query string (definition terms + original question)."""
    if not analysis.sparse_terms:
        return analysis.retrieval_query
    return " ".join([analysis.retrieval_query, *analysis.sparse_terms])
