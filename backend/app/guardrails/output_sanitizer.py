"""Sanitize LLM output — strip chat-template leakage and forbidden injury advice."""

from __future__ import annotations

import re

# Groq/Llama chat template tokens that must never appear in user-facing output
_TEMPLATE_LEAK_RE = re.compile(
    r"<\|[^|>]*\|>|"
    r"<\|redacted_[^>]+\|>|"
    r"<\|start_header_id\|>|<\|end_header_id\|>|"
    r"<\|eot_id\|>|<\|begin_of_text\|>",
    re.I,
)

# "improve chest deflection to ≥ 50 mm" class errors
_INJURY_INCREASE_RE = re.compile(
    r"(?:increase|raise|improve|higher|worsen|exceed)\s+.{0,40}"
    r"(?:chest\s+deflection|hic|head\s+injury|pelvis|femur|neck\s+injury|"
    r"thorax|abdomen|injury\s+criterion)",
    re.I,
)

_INJURY_INCREASE_RE2 = re.compile(
    r"(?:chest\s+deflection|hic|head\s+injury).{0,30}"
    r"(?:≥|>=|to at least|to a minimum of)\s*\d+",
    re.I,
)


def sanitize_model_output(text: str) -> tuple[str, list[str]]:
    """
    Returns (cleaned_text, warnings).
    Replaces forbidden injury-increase advice with a safety notice.
    """
    warnings: list[str] = []
    out = text or ""

    if _TEMPLATE_LEAK_RE.search(out):
        out = _TEMPLATE_LEAK_RE.sub("", out)
        warnings.append("template_leak_stripped")

    if _INJURY_INCREASE_RE.search(out) or _INJURY_INCREASE_RE2.search(out):
        warnings.append("injury_increase_blocked")
        out = (
            "**Safety notice:** Passive safety engineering requires *reducing* occupant "
            "injury metrics (chest deflection, HIC, etc.), not increasing them. "
            "The retrieved context does not support recommending higher injury values.\n\n"
            + re.sub(
                _INJURY_INCREASE_RE, "[removed — cannot recommend increasing injury values]",
                out,
            )
        )
        out = _INJURY_INCREASE_RE2.sub(
            "[removed — cannot recommend increasing injury values]", out
        )

    # Collapse excessive blank lines after stripping
    out = re.sub(r"\n{4,}", "\n\n\n", out).strip()
    return out, warnings


def contains_injury_increase_advice(text: str) -> bool:
    return bool(_INJURY_INCREASE_RE.search(text) or _INJURY_INCREASE_RE2.search(text))
