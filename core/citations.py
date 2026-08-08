"""Citation mapping + grouped LLM context assembly."""

from __future__ import annotations

import re
from typing import Any

from app.config import settings

# Standard [S1], CJK 【S1】, parenthetical (S1), bare "S1" after punctuation.
_CITE_PATTERNS = (
    re.compile(r"\[S(\d+)\]", re.I),
    re.compile(r"【S(\d+)】", re.I),
    re.compile(r"「S(\d+)」", re.I),
    re.compile(r"\(S(\d+)\)", re.I),
    re.compile(r"(?<=[\s,.:;(])S(\d+)(?=[\s,.:;)\]】]|$)", re.I),
)


def compress_chunk_text(text: str, max_chars: int | None = None) -> str:
    """Context compression — trim long chunks while keeping head/tail."""
    limit = max_chars or settings.MAX_CHUNK_CHARS
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n…\n" + text[-half:]


def normalize_citation_marks(text: str) -> str:
    """Map common LLM citation variants to canonical [S#] form for matching."""
    out = text or ""
    out = re.sub(r"【S(\d+)】", r"[S\1]", out, flags=re.I)
    out = re.sub(r"「S(\d+)」", r"[S\1]", out, flags=re.I)
    return out


_CITATIONS_SECTION_RE = re.compile(
    r"\nCitations:\s*\n(.*?)(?:\n\n|\Z)",
    re.I | re.S,
)
_STRUCTURED_CITE_LINE = re.compile(
    r"^\s*(?:[-*]|\d+\.)\s*(.+?)\s*[-–—:]\s*\[?S(\d+)\]?\.?\s*$",
    re.I | re.M,
)
_STRUCTURED_CITE_INLINE = re.compile(r"\[S(\d+)\]", re.I)


def _answer_body_without_citations_section(answer: str) -> str:
    """Strip trailing Citations: block for inline extraction."""
    text = answer or ""
    match = _CITATIONS_SECTION_RE.search(text)
    if match:
        return text[: match.start()].strip()
    return text


def extract_structured_citation_ids(answer: str) -> set[str]:
    """Parse dedicated Citations: section at end of answer."""
    match = _CITATIONS_SECTION_RE.search(answer or "")
    if not match:
        return set()
    block = match.group(1)
    found: set[str] = set()
    for m in _STRUCTURED_CITE_LINE.finditer(block):
        found.add(f"S{m.group(2)}")
    for num in _STRUCTURED_CITE_INLINE.findall(block):
        found.add(f"S{num}")
    return found


def extract_citation_ids(answer: str) -> set[str]:
    """Collect S# IDs from structured Citations: block and inline tags."""
    structured = extract_structured_citation_ids(answer)
    body = _answer_body_without_citations_section(answer)
    normalized = normalize_citation_marks(body)
    inline: set[str] = set()
    for pattern in _CITE_PATTERNS:
        for num in pattern.findall(normalized):
            inline.add(f"S{num}")
    return structured | inline


def build_grouped_context(chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Group chunks by regulation; assign stable S# IDs tied to chunk metadata."""
    if not chunks:
        return "", []

    groups: dict[str, list[dict[str, Any]]] = {}
    reg_order: list[str] = []
    for chunk in chunks:
        reg = chunk.get("regulation_code") or "UNKNOWN"
        if reg not in groups:
            groups[reg] = []
            reg_order.append(reg)
        groups[reg].append(chunk)

    lines: list[str] = []
    enriched: list[dict[str, Any]] = []
    cite_idx = 1

    for reg in reg_order:
        reg_chunks = groups[reg]
        title = reg_chunks[0].get("title") or reg
        lines.append(f"=== {reg} | {title} ===")

        for chunk in reg_chunks:
            citation_id = f"S{cite_idx}"
            body = compress_chunk_text(chunk.get("chunk_text") or "")
            tagged = {
                **chunk,
                "citation_id": citation_id,
                "citation_label": (
                    f"{reg} | {chunk.get('document_name', '')} "
                    f"p.{chunk.get('page_number', '?')} "
                    f"§{chunk.get('section') or 'General'}"
                ),
            }
            enriched.append(tagged)
            lines.append(f"[{citation_id}] {tagged['citation_label']}\n{body}\n")
            cite_idx += 1
        lines.append("")

    return "\n".join(lines).strip(), enriched


def build_citations_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        citation_id = chunk.get("citation_id") or f"S{idx}"
        citations.append(
            {
                "id": citation_id,
                "chunk_id": chunk.get("chunk_id"),
                "regulation_code": chunk.get("regulation_code"),
                "document_name": chunk.get("document_name"),
                "page_number": chunk.get("page_number"),
                "section": chunk.get("section"),
                "title": chunk.get("title"),
                "snippet": (chunk.get("chunk_text") or "")[:280],
                "score": chunk.get("score"),
                "rrf_score": chunk.get("rrf_score"),
            }
        )
    return citations


def mark_citations_referenced_in_answer(
    citations: list[dict[str, Any]], answer: str
) -> list[dict[str, Any]]:
    refs = extract_citation_ids(answer)
    return [{**cite, "referenced_in_answer": cite["id"] in refs} for cite in citations]
