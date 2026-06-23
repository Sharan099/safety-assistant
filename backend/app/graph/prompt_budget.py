"""Prompt token budget helpers — strip redundant chunk boilerplate for the LLM."""

from __future__ import annotations

import re

_CHUNK_HEADER_LINE_RE = re.compile(
    r"^\[[A-Z0-9_]+\s*\|[^\]]+\]\s*\n?",
    re.I | re.M,
)
_HEADING_PATH_LINE_RE = re.compile(
    r"^\[[^\]]+>\s*[^\]]+\]\s*\n?",
    re.M,
)
_HASH_HEADING_RE = re.compile(r"^#\s+.+?\n+", re.M)


def strip_chunk_boilerplate(text: str) -> str:
    """
    Remove duplicate structural prefixes already captured in citation labels.
    Keeps APPLICABILITY blocks — still required for category verification.
    """
    body = (text or "").strip()
    for _ in range(3):
        prev = body
        body = _CHUNK_HEADER_LINE_RE.sub("", body, count=1)
        body = _HEADING_PATH_LINE_RE.sub("", body, count=1)
        body = _HASH_HEADING_RE.sub("", body, count=1)
        body = body.strip()
        if body == prev:
            break
    return body


def estimate_tokens(text: str) -> int:
    """Rough token estimate (words × 1.33) — matches gateway classifier."""
    return max(1, int(len((text or "").split()) * 1.33))


def passage_char_budget(max_context_tokens: int, passage_count: int, overhead_tokens: int) -> int:
    """Per-passage char budget from total context token ceiling."""
    if passage_count <= 0:
        return 700
    available = max(200, max_context_tokens - overhead_tokens)
    per_passage_tokens = max(80, available // passage_count)
    return int(per_passage_tokens * 3.5)  # ~3.5 chars per token for dense legal text
