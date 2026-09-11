"""Pure relevance / authority / scope filters — no DB, no model.

`is_relevant` is a literal-vocabulary floor applied after ranking: a chunk
must share enough significant terms with the query. Dense similarity alone
can surface topically unrelated text; this guard is independent of embedding
quality (kept from the baseline, where it was motivated by reproduced
failures).
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "with",
        "by",
        "at",
        "from",
        "this",
        "that",
        "it",
        "as",
        "what",
        "which",
        "who",
        "does",
        "do",
        "did",
        "how",
        "why",
        "when",
        "where",
        "will",
        "would",
        "should",
        "can",
        "could",
        "not",
    }
)

KNOWN_AUTHORITY_LEVELS = frozenset(
    {"AUTHORITATIVE", "OFFICIAL_DOCUMENTATION", "INTERNAL_APPROVED", "HISTORICAL", "REFERENCE", "SYNTHETIC"}
)
DEFAULT_MIN_SHARED_TERMS = 2


def significant_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def shared_term_count(query_text: str, content: str) -> int:
    return len(significant_tokens(query_text) & significant_tokens(content))


def is_relevant(query_text: str, content: str, *, min_shared_terms: int = DEFAULT_MIN_SHARED_TERMS) -> bool:
    query_terms = significant_tokens(query_text)
    if not query_terms:
        return True
    return shared_term_count(query_text, content) >= min(min_shared_terms, len(query_terms))


def has_known_authority(authority_level: str) -> bool:
    return authority_level in KNOWN_AUTHORITY_LEVELS


@dataclass(frozen=True)
class ScopeFilter:
    """Deterministic pre-ranking scope. Applied in SQL before any scoring.

    - ``as_of``: temporal validity date. ``None`` = "currently effective".
    - ``regulation_keys``: restrict to specific regulations (exact-lookup routes).
    - ``kinds``/``authority_levels``: metadata restrictions.
    - ``include_superseded``: historical/comparison queries opt in explicitly.
    - ``data_classes``: what the principal is authorised to see (M11).
    """

    as_of: datetime.date | None = None
    regulation_keys: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    authority_levels: tuple[str, ...] = ()
    include_superseded: bool = False
    data_classes: tuple[str, ...] = ("PUBLIC",)
    version_ids: tuple[str, ...] = field(default=())

    def effective_date(self, today: datetime.date | None = None) -> datetime.date:
        return self.as_of or today or datetime.date.today()
