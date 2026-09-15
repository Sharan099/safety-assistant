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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safety_assistant.retrieval.authz import Authz

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


def light_stem(token: str) -> str:
    """Deterministic English suffix stripping (no dictionary): doors→door,
    categories→category, applies→apply, exceeded→exceed. Deliberately
    conservative — identifiers and short tokens are left alone."""
    if len(token) <= 4 or not token.isalpha():
        return token
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)] + repl
            return stem if suffix != "s" or not token.endswith("ss") else token
    return token


DEFINITION_INTENT = re.compile(r"\b(defin(e|ed|es|ition|itions)|meaning of|what (is|are)|what does .* mean)\b", re.I)
_DEFINITION_STOP = {
    *"definition definitions define defined defines meaning mean means what does term".split(),
    *"according accordance under per regulation regulations un ece".split(),
}


_HYPHEN_WORD_RE = re.compile(r"[a-z0-9*]+(?:-[a-z0-9*]+)*")


def definition_terms(query: str) -> list[str]:
    """Content terms of a definition question ("what is X" → the words of X) in query order, at most
    four. Hyphenated names stay whole ("i-size", "r-point") — the defined phrase is quoted verbatim."""
    out: list[str] = []
    for w in _HYPHEN_WORD_RE.findall(query.lower()):
        t = light_stem(w)
        if (
            len(t) > 2
            and w not in _STOPWORDS
            and w not in _DEFINITION_STOP
            and t not in out
            and not re.fullmatch(r"r\d+", w)
        ):
            out.append(t)
    return out[:4]


def defined_phrase_pattern(terms: list[str]) -> str:
    """Regex for the quoted defined phrase made of `terms` in order, each allowed a suffix — the
    stemmed "lock" still matches '"Emergency locking retractor" means'. Case-insensitive at use."""
    return '"' + r"\W+".join(re.escape(t) + r"\w*" for t in terms) + r'[^"]{0,4}"'


def defines_term(query: str, content: str) -> bool:
    """True when `content` is the regulation's own definition of the thing the question asks about:
    a definition-intent query whose content terms form a quoted defined phrase ('"ISOFIX" means')."""
    terms = definition_terms(query)
    return bool(terms) and re.search(defined_phrase_pattern(terms), content, re.I) is not None


# Engineering words → the regulation's own vocabulary, appended for lexical retrieval only.
SYNONYMS = {
    "webbing": "strap",
    "seatbelt": "safety-belt",
    "seatbelts": "safety-belts",
    "bumper": "protective device",
    "childseat": "child restraint",
    "windscreen": "windshield",
    "pedestrian head": "headform HIC",
    "head impact": "headform HIC",
}


def expand_synonyms(query: str) -> str:
    extra = [v for k, v in SYNONYMS.items() if re.search(rf"\b{k}\b", query, re.IGNORECASE)]
    return f"{query} {' '.join(extra)}" if extra else query


def significant_tokens(text: str) -> set[str]:
    return {light_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def shared_term_count(query_text: str, content: str) -> int:
    return len(significant_tokens(query_text) & significant_tokens(content))


def is_relevant(query_text: str, content: str, *, min_shared_terms: int = DEFAULT_MIN_SHARED_TERMS) -> bool:
    query_terms = significant_tokens(query_text)
    if not query_terms:
        return True
    # Very short queries ("ISOFIX definition") carry one real term; demand one match, not two.
    required = 1 if len(query_terms) <= 2 else min(min_shared_terms, len(query_terms))
    return shared_term_count(query_text, content) >= required


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
    - ``authz``: document-level ownership/scope predicate (ADR-0029 §4). ``None`` = no identity:
      authoritative documents only.
    """

    as_of: datetime.date | None = None
    regulation_keys: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    authority_levels: tuple[str, ...] = ()
    include_superseded: bool = False
    data_classes: tuple[str, ...] = ("PUBLIC",)
    version_ids: tuple[str, ...] = field(default=())
    authz: Authz | None = None

    def effective_date(self, today: datetime.date | None = None) -> datetime.date:
        return self.as_of or today or datetime.date.today()
