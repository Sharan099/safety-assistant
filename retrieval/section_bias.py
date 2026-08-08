"""Section-category bias for requirements / installation / scope / definitions.

Uses existing chunk metadata (``section_number``, ``section_title``,
``content_type``) as a soft retrieval boost — not a hard Qdrant filter —
so hybrid recall stays intact while matching section types float up.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

logger = logging.getLogger(__name__)

SectionCategory = str  # scope | definitions | requirements | installation

_CATEGORY_CUES: list[tuple[SectionCategory, re.Pattern[str]]] = [
    (
        "definitions",
        re.compile(
            r"(?ix)\b("
            r"define\b|definition\s+of|definitions\b|what\s+does\s+\w+\s+mean|"
            r"meaning\s+of\b"
            r")\b"
        ),
    ),
    (
        "scope",
        re.compile(
            r"(?ix)\b("
            r"scope\b|objective\b|purpose\s+of\b|what\s+(?:is|are)\s+this\s+regulation\s+about|"
            r"applicability\b|vehicles?\s+(?:are\s+)?covered\b|covered\s+under\b"
            r")\b"
        ),
    ),
    (
        "installation",
        re.compile(
            r"(?ix)\b("
            r"installation\b|install(?:ed|ing)?\b|fitting\b|fitted\b|"
            r"how\s+(?:to|is|are).{0,40}\binstall"
            r")\b"
        ),
    ),
    (
        "requirements",
        re.compile(
            r"(?ix)\b("
            r"requirements?\b|specifications?\b|performance\s+criteria\b|"
            r"shall\s+(?:not\s+)?(?:exceed|comply)|injury\s+criteria\b|"
            r"pass\s*/\s*fail\s+criteria\b"
            r")\b"
        ),
    ),
]

# UNECE drafting conventions commonly used in this corpus.
_SECTION_TITLE_HINTS: dict[SectionCategory, tuple[str, ...]] = {
    "scope": ("scope", "purpose", "application"),
    "definitions": ("definition", "definitions"),
    "requirements": (
        "requirement",
        "specification",
        "performance",
        "criteria",
        "general specifications",
    ),
    "installation": ("installation", "install", "fitting"),
}

# Clause-number heuristics (UNECE R94/R95 style): 1=Scope, 2=Definitions, 5=Requirements.
_SECTION_PREFIX: dict[SectionCategory, tuple[str, ...]] = {
    "scope": ("1", "1."),
    "definitions": ("2", "2."),
    "requirements": ("5", "5."),
    "installation": ("6", "6.", "7", "7."),  # R16 installation-ish; soft only
}


def detect_section_category(question: str) -> SectionCategory | None:
    q = (question or "").strip()
    if not q:
        return None
    for cat, pat in _CATEGORY_CUES:
        if pat.search(q):
            return cat
    return None


def _section_match_score(chunk: object, category: SectionCategory) -> float:
    sec = str(getattr(chunk, "section_number", "") or "").strip()
    title = str(getattr(chunk, "section_title", "") or "").lower()
    text_head = str(getattr(chunk, "text", "") or "")[:240].lower()
    ctype = str(getattr(chunk, "content_type", "") or "").lower()
    score = 0.0

    for prefix in _SECTION_PREFIX.get(category, ()):
        if sec == prefix.rstrip(".") or sec.startswith(prefix):
            score += 2.0
            break

    for hint in _SECTION_TITLE_HINTS.get(category, ()):
        if hint in title or hint in text_head:
            score += 1.5
            break

    if category == "requirements" and ctype == "table":
        score += 0.5
    if category == "requirements" and ctype == "figure":
        score += 0.75
    return score


_FIGURE_QUERY_RE = re.compile(
    r"(?ix)\b("
    r"figure\s+\d+|force-?time(?:\s+performance)?(?:\s+curve)?|"
    r"performance\s+curve|shown\s+in\s+figure"
    r")\b"
)


def bias_chunks_by_section_category(
    chunks: Sequence[object],
    question: str,
    *,
    category: SectionCategory | None = None,
) -> list:
    """Soft-boost chunks whose section metadata matches the query category."""
    cat = category or detect_section_category(question)
    fig_query = bool(_FIGURE_QUERY_RE.search(question or ""))

    if not cat and not fig_query:
        return list(chunks)
    if not chunks:
        return list(chunks)

    scored: list[tuple[float, int, object]] = []
    for i, c in enumerate(chunks):
        base = float(getattr(c, "score", 0.0) or 0.0)
        bonus = _section_match_score(c, cat) if cat else 0.0
        if fig_query and str(getattr(c, "content_type", "") or "").lower() == "figure":
            bonus += 2.0
        scored.append((base + bonus, -i, c))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    out = [c for _s, _i, c in scored]
    if cat and out and out[0] is not chunks[0]:
        logger.info(
            "section_category_bias category=%s top_section=%s",
            cat,
            getattr(out[0], "section_number", None),
        )
    return out
