"""Honest-refusal detection for RAGAS eval (manual override column)."""

from __future__ import annotations

import re

_REFUSAL_PHRASES = (
    "couldn't find",
    "could not find",
    "cannot find",
    "can't find",
    "not found",
    "not in the",
    "not available",
    "do not contain",
    "does not contain",
    "no relevant",
    "not present in the sources",
    "not present in the",
    "i'm sorry, but",
    "i am sorry, but",
)

_NOT_IN_CORPUS_RE = re.compile(
    r"\b(?:R\s*)?999\b|nonexistent|not in the ingested|not in (?:the )?corpus",
    re.I,
)


def expect_refusal(case: dict) -> bool:
    return bool(case.get("expect_refusal"))


def refusal_is_correct(case: dict, answer: str, *, evidence_only: bool = False) -> bool | None:
    """True/false when ground truth expects refusal; None otherwise."""
    if not expect_refusal(case):
        return None
    text = (answer or "").strip().lower()
    if not text and evidence_only:
        return True
    if any(p in text for p in _REFUSAL_PHRASES):
        return True
    if _NOT_IN_CORPUS_RE.search(case.get("ground_truth", "")) and _NOT_IN_CORPUS_RE.search(
        answer or ""
    ):
        return True
    # Hallucinated numeric limit or unrelated regulation citation
    if re.search(r"\b\d+\s*mm\b", answer or "", re.I) and "999" in case.get("question", ""):
        return False
    if re.search(r"\bUN[\s_-]?R?\d+", answer or "", re.I) and "999" in case.get("question", ""):
        if "999" not in answer:
            return False
    return len(text) < 20 or "sorry" in text
