"""Relevance/authority guard — pure functions, no dependency on
`packages.retrieval.search` (kept separate to avoid a circular import;
`search.py` applies these plus dedup, which needs the result list itself).

CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 8 / PRD_COPILOT_UPDATE.md §8:
"Do not expose raw top-k chunks." Motivated by two real, reproduced
problems, not a hypothetical:

1. `full_text_search`'s previous `plainto_tsquery` ANDs every query term
   together, so a realistic multi-term investigation query ("restraint
   configuration belt force limiter webbing revision") returned ZERO
   full-text hits even though the corpus discusses belts and force
   limiters extensively — fixed in `search.py` by OR-joining tokenized
   query terms into an explicit `to_tsquery` instead.
2. The interim `HashingEmbeddingProvider` (docs/ADR/0007) has no semantic
   understanding, so vector search alone can surface a chunk sharing no
   real topic with the query but scoring nonzero via hash collisions —
   `is_relevant` below is a lexical relevance floor independent of
   embedding quality: the query and the chunk must share enough literal
   vocabulary, on top of RRF ranking.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately short — excluded from the overlap count so two chunks don't
# "match" on "the"/"and"/"of" alone.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
        "was", "were", "be", "been", "with", "by", "at", "from", "this", "that",
        "it", "as", "what", "which", "who", "does", "do", "did", "how", "why",
        "when", "where", "will", "would", "should", "can", "could", "not",
    }
)  # fmt: skip

# BACKEND_SCHEMA.md §18 KnowledgeSource.source_type / authority_level values.
KNOWN_AUTHORITY_LEVELS = frozenset(
    {"AUTHORITATIVE", "OFFICIAL_DOCUMENTATION", "INTERNAL_APPROVED", "HISTORICAL", "REFERENCE", "SYNTHETIC"}
)

DEFAULT_MIN_SHARED_TERMS = 2


def significant_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def shared_term_count(query_text: str, content: str) -> int:
    return len(significant_tokens(query_text) & significant_tokens(content))


def is_relevant(query_text: str, content: str, *, min_shared_terms: int = DEFAULT_MIN_SHARED_TERMS) -> bool:
    """The chunk must share at least `min_shared_terms` significant terms
    with the query (or every term the query has, if it has fewer than
    that). A trivial/empty query passes everything through rather than
    rejecting on nothing to check against."""
    query_terms = significant_tokens(query_text)
    if not query_terms:
        return True
    required = min(min_shared_terms, len(query_terms))
    return shared_term_count(query_text, content) >= required


def has_known_authority(authority_level: str) -> bool:
    return authority_level in KNOWN_AUTHORITY_LEVELS
