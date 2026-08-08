"""Grounding verification — citation validation and answer confidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.citations import extract_citation_ids, extract_structured_citation_ids

_CITE_RE = re.compile(r"\[S(\d+)\]", re.I)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Regulatory fact patterns that should carry a citation.
_FACT_RE = re.compile(
    r"\b(shall|must|required|limit|maximum|minimum|criteria|HIC|ThCC|kN|mm)\b",
    re.I,
)


@dataclass
class GroundingReport:
    confidence: float
    cited_ids: list[str]
    invalid_citations: list[str]
    unsupported_sentences: list[str]
    grounded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": round(self.confidence, 3),
            "cited_ids": self.cited_ids,
            "invalid_citations": self.invalid_citations,
            "unsupported_sentences": self.unsupported_sentences[:5],
            "grounded": self.grounded,
        }


def verify_grounding(answer: str, citations: list[dict[str, Any]]) -> GroundingReport:
    """Check citation IDs exist and flag likely unsupported regulatory claims.

    With STRUCTURAL_CITATIONS, claims may omit inline [S#] and list IDs in a
    trailing ``Citations:`` section — that is valid. Per-sentence unsupported
    checks only apply when there is no structured Citations block.
    """
    valid_ids = {c["id"] for c in citations if c.get("id")}
    cited = sorted(extract_citation_ids(answer or ""))
    invalid = [cid for cid in cited if cid not in valid_ids]

    structured = extract_structured_citation_ids(answer or "")
    unsupported: list[str] = []
    # Structured Citations: section covers claims by design — do not penalize
    # missing inline [S#] on body sentences (frontier_test refusal false positive).
    if not structured:
        sentences = [s.strip() for s in _SENTENCE_RE.split(answer or "") if s.strip()]
        for sentence in sentences:
            if _FACT_RE.search(sentence) and not _CITE_RE.search(sentence):
                if not sentence.lower().startswith(("i could not", "i cannot", "the question")):
                    unsupported.append(sentence[:200])

    cite_ratio = len(set(cited)) / max(len(valid_ids), 1)
    penalty = 0.2 * len(invalid) + 0.15 * len(unsupported)
    confidence = max(0.0, min(1.0, min(cite_ratio, 1.0) - penalty))
    grounded = confidence >= 0.25 and not invalid

    return GroundingReport(
        confidence=confidence,
        cited_ids=cited,
        invalid_citations=invalid,
        unsupported_sentences=unsupported,
        grounded=grounded,
    )


def apply_low_confidence_prefix(report: GroundingReport, answer: str) -> str:
    """Prepend caution when grounding confidence is low (user display only).

    Hard refusal (canned 'I could not produce a reliable answer') lives in
    core/search.py and fires only when ALL of:
      - answer was NOT truncated at the output token cap
      - confidence < GROUNDING_MIN_CONFIDENCE
      - unsupported_sentences is non-empty
      - not evidence_only
    Low confidence alone (unsupported_sentences empty) → disclaimer prefix only.
    """
    if report.grounded:
        return answer
    return (
        "Note: answer confidence is limited — verify against the cited sources.\n\n"
        + answer
    )
