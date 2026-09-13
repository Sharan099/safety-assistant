"""Lexical + authority heuristic reranker — the production default.

Score = fused RRF score + a small bonus for literal query-term overlap
(rewards exact identifiers like "HIC15", "5.2.1.8") + a small authority-tier
bonus. Measured on the pre-rebuild 8-case set: lifts MRR 0.667 → 0.900 over
RRF alone on the baseline golden set. The cross-encoder is better still on
nDCG but ~3.5 s/query on CPU (docs/ADR/0015) — opt-in via RERANKER=cross_encoder.
"""

from __future__ import annotations

import re

from safety_assistant.providers.rerankers.base import RerankCandidate
from safety_assistant.retrieval.filters import significant_tokens

_AUTHORITY_BONUS = {
    "AUTHORITATIVE": 0.02,
    "OFFICIAL_DOCUMENTATION": 0.015,
    "INTERNAL_APPROVED": 0.01,
    "HISTORICAL": 0.005,
    "REFERENCE": 0.0,
    "SYNTHETIC": 0.0,
}
_MAX_OVERLAP_TERMS_COUNTED = 5
_OVERLAP_BONUS_PER_TERM = 0.01
# Requirement-seeking queries prefer normative clauses over annex procedures
# describing how a criterion is computed.
_REQUIREMENT_CUE_RE = re.compile(
    r"(limit|limits|maximum|minimum|exceed|shall|must|allowed|permitted|require|requirement"
    r"|criterion value|at most|at least)",
    re.IGNORECASE,
)
_NORMATIVE_BONUS = 0.008


class LexicalAuthorityReranker:
    model_name = "lexical-authority-heuristic"
    model_version = "v1"

    def score(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        query_terms = significant_tokens(query)
        wants_requirement = bool(_REQUIREMENT_CUE_RE.search(query))
        out = []
        for c in candidates:
            overlap = len(query_terms & significant_tokens(c.content))
            score = (
                c.fused_score
                + min(overlap, _MAX_OVERLAP_TERMS_COUNTED) * _OVERLAP_BONUS_PER_TERM
                + _AUTHORITY_BONUS.get(c.authority_level, 0.0)
            )
            if wants_requirement and c.normative:
                score += _NORMATIVE_BONUS
            out.append(score)
        return out
