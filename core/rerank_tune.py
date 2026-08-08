"""Post-rerank tuning to preserve short definition chunks."""

from __future__ import annotations

import re
from typing import Any

_DEFINITION_MARKERS = [
    re.compile(rf"\b{pat}\b", re.I)
    for pat in (
        r"means",
        r"shall mean",
        r"is defined as",
        r"definition[s]?",
        r"for the purposes of",
    )
]

_DEFINITION_SECTION_RE = re.compile(r"^(?:2(?:\.\d+)?|definitions)\b", re.I)
_SHORT_CHUNK_CHARS = 800
_SHORT_BOOST = 0.08
_DEFINITION_BOOST = 0.12
_SECTION_BOOST = 0.10


def _definition_bonus(chunk: dict[str, Any]) -> float:
    text = chunk.get("chunk_text") or ""
    section = (chunk.get("section") or "").strip()
    bonus = 0.0

    if len(text) <= _SHORT_CHUNK_CHARS:
        bonus += _SHORT_BOOST

    if any(pat.search(text) for pat in _DEFINITION_MARKERS):
        bonus += _DEFINITION_BOOST

    if _DEFINITION_SECTION_RE.match(section) or "definition" in section.lower():
        bonus += _SECTION_BOOST

    return bonus


def boost_definition_chunks(
    chunks: list[dict[str, Any]], *, enabled: bool
) -> list[dict[str, Any]]:
    """Re-rank fused list to keep concise definition passages from being demoted."""
    if not enabled or not chunks:
        return chunks

    tuned: list[dict[str, Any]] = []
    for chunk in chunks:
        base = float(chunk.get("score", chunk.get("rrf_score", 0.0)))
        item = {**chunk, "score": base + _definition_bonus(chunk)}
        tuned.append(item)

    tuned.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return tuned
