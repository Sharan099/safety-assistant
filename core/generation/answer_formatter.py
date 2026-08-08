"""Post-generation answer cleanup for display and evaluation scoring."""

from __future__ import annotations

import re

from core.generation.llm_output_cleaner import clean_llm_output
from core.grounding import GroundingReport

# Display-only prefixes that hurt RAGAS relevancy/correctness if left in scored text.
_SCORING_STRIP_PATTERNS = (
    re.compile(
        r"^Note:\s*answer confidence is limited[^\n]*\n+",
        re.I,
    ),
    re.compile(
        r"^According to the (?:provided |retrieved )?(?:context|sources|documents)[^\n]*\n+",
        re.I,
    ),
    re.compile(
        r"^Based on the (?:provided |retrieved )?(?:context|sources|documents)[^\n]*\n+",
        re.I,
    ),
    re.compile(
        r"^In the (?:provided |retrieved )?(?:context|sources|documents)[^\n]*\n+",
        re.I,
    ),
)

_HARD_REFUSAL_RE = re.compile(
    r"^I could not produce a reliable answer from the retrieved UNECE sources\.",
    re.I,
)


def strip_scoring_artifacts(answer: str) -> str:
    """Remove reasoning blocks and display-only boilerplate before RAGAS scoring."""
    text = clean_llm_output(answer)
    for pattern in _SCORING_STRIP_PATTERNS:
        text = pattern.sub("", text).strip()
    return text


def format_answer_for_display(report: GroundingReport, answer: str) -> str:
    """Return answer for API users — no confidence disclaimer unless grounding failed entirely."""
    text = clean_llm_output(answer)
    if _HARD_REFUSAL_RE.match(text):
        return text
    # Grounded answers pass through unchanged (no "confidence is limited" prefix).
    return text
