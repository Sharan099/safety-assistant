"""User-safe generation error handling — no raw provider text to clients."""

from __future__ import annotations

import re

GENERATION_UNAVAILABLE_MESSAGE = (
    "The system is temporarily unable to answer — please retry in a moment."
)

_RAW_ERROR_MARKERS = (
    "llm error:",
    "rate limit",
    "rate_limit",
    "429",
    "502",
    "503",
    "504",
    "timeout",
    "overloaded",
    "api status",
    "bad gateway",
    "service unavailable",
    "upsell",
    "upgrade your plan",
    "tokens per minute",
    "requests per minute",
)

_RAW_ERROR_RE = re.compile(
    r"(error code:\s*\d{3}|status code:\s*\d{3}|httpx\.\w+|"
    r"groq\.\w+error|anthropic\.\w+error|providererror)",
    re.I,
)


def is_raw_provider_error(text: str | None) -> bool:
    """True when answer text looks like a leaked provider / HTTP error."""
    if not text:
        return False
    low = text.lower().strip()
    if low.startswith("error:") and any(m in low for m in _RAW_ERROR_MARKERS):
        return True
    if low.startswith("llm error:"):
        return True
    if _RAW_ERROR_RE.search(text):
        return True
    return any(m in low for m in _RAW_ERROR_MARKERS) and (
        "please retry" not in low or "temporarily unable" not in low
    )


def sanitize_user_answer(text: str | None, *, generation_failed: bool = False) -> str:
    if generation_failed or is_raw_provider_error(text):
        return GENERATION_UNAVAILABLE_MESSAGE
    return (text or "").strip() or GENERATION_UNAVAILABLE_MESSAGE
