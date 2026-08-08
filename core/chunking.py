"""Adaptive chunking with optional parent-child section summaries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import settings
from core.documents import load_pages

APPROX_CHARS_PER_TOKEN = 4
MIN_TOKENS = 100
MAX_TOKENS = 1100

_INJURY_LIMIT_CLAUSE = re.compile(
    r"((?:\d+\.)+\d+\.)\s*\n?\s*((?:The\s+)?[^\n]{8,200}?shall not exceed\s+[^;\n]+;)",
    re.I | re.MULTILINE,
)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // APPROX_CHARS_PER_TOKEN)


def parse_pdf(path: Path) -> list[dict[str, Any]]:
    return load_pages(path)


def _recursive_split(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    separators = separators or ["\n\n", "\n", ". ", " "]
    target_chars = chunk_size * APPROX_CHARS_PER_TOKEN
    overlap_chars = overlap * APPROX_CHARS_PER_TOKEN
    if len(text) <= target_chars:
        return [text] if text.strip() else []

    sep = separators[0] if separators else ""
    rest = separators[1:] if len(separators) > 1 else [""]
    parts = text.split(sep) if sep else [text]
    chunks: list[str] = []
    current = ""

    for part in parts:
        piece = part if not sep else part + sep
        if len(current) + len(piece) <= target_chars:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        if len(piece) > target_chars and rest:
            chunks.extend(_recursive_split(piece, chunk_size, overlap, rest))
            current = ""
        else:
            current = piece

    if current.strip():
        chunks.append(current.strip())

    if overlap_chars > 0 and len(chunks) > 1:
        merged: list[str] = [chunks[0]]
        for nxt in chunks[1:]:
            prev = merged[-1]
            tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            merged.append((tail + " " + nxt).strip())
        chunks = merged

    return [c for c in chunks if c.strip()]


def _page_split(pages: list[dict[str, Any]], chunk_size: int, overlap: int) -> list[str]:
    raw = [p["text"] for p in pages if p.get("text")]
    merged: list[str] = []
    for page_text in raw:
        if _approx_tokens(page_text) > MAX_TOKENS:
            merged.extend(_recursive_split(page_text, chunk_size, overlap))
        elif page_text.strip():
            merged.append(page_text.strip())
    return merged


def _size_compliance(chunks: list[str]) -> float:
    if not chunks:
        return 0.0
    ok = sum(1 for c in chunks if MIN_TOKENS <= _approx_tokens(c) <= MAX_TOKENS)
    return ok / len(chunks)


def _block_integrity(chunks: list[str], source: str) -> float:
    """Fraction of paragraph blocks kept intact inside a single chunk."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", source) if b.strip()]
    if not blocks:
        return 1.0
    intact = 0
    for block in blocks:
        if len(block) < 40:
            intact += 1
            continue
        for chunk in chunks:
            if block in chunk:
                intact += 1
                break
    return intact / len(blocks)


def _mean_score(chunks: list[str], source: str) -> float:
    sc = _size_compliance(chunks)
    bi = _block_integrity(chunks, source)
    return 0.5 * sc + 0.5 * bi


def _page_for_offset(pages: list[dict[str, Any]], offset: int) -> int:
    running = 0
    for page in pages:
        block = page["text"] + "\n\n"
        running += len(block)
        if offset < running:
            return page["page_number"]
    return pages[-1]["page_number"] if pages else 1


def _page_for_chunk(pages: list[dict[str, Any]], chunk_text: str) -> int:
    """Assign page by best text overlap (offset tracking breaks under overlap/merge)."""
    if not pages:
        return 1
    # Prefer a mid-window probe so headers/footers don't dominate.
    cleaned = re.sub(r"\s+", " ", chunk_text or "").strip()
    if len(cleaned) < 20:
        return pages[0]["page_number"]
    start = max(0, len(cleaned) // 4)
    probe = cleaned[start : start + 80].lower()
    if len(probe) < 20:
        probe = cleaned[:80].lower()
    best_page = pages[0]["page_number"]
    best_score = -1
    for page in pages:
        hay = re.sub(r"\s+", " ", page.get("text") or "").lower()
        if not hay:
            continue
        if probe in hay:
            return page["page_number"]
        # Fallback: shared token overlap on a short window
        score = sum(1 for tok in probe.split() if len(tok) > 3 and tok in hay)
        if score > best_score:
            best_score = score
            best_page = page["page_number"]
    return best_page


def _detect_section(text: str) -> str:
    annex = re.search(r"\b(Annex\s+\d+)\b", text, re.I)
    if annex:
        return annex.group(1).title()
    sec = re.search(r"\b((?:\d+\.)+\d+)\b", text)
    return sec.group(1) if sec else "General"


def parse_pdf(path: Path) -> list[dict[str, Any]]:
    return load_pages(path)


def _parent_child_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add one parent summary chunk per section to help retrieval of short definitions."""
    if not settings.PARENT_CHILD_CHUNKING:
        return chunks

    by_section: dict[str, list[dict[str, Any]]] = {}
    for c in chunks:
        by_section.setdefault(c.get("section") or "General", []).append(c)

    out: list[dict[str, Any]] = []
    parent_index = 0
    for section, section_chunks in by_section.items():
        combined = "\n".join(c["chunk_text"] for c in section_chunks)
        summary = combined[:600].strip()
        if summary:
            out.append(
                {
                    "chunk_text": f"[Section summary §{section}]\n{summary}",
                    "chunk_index": parent_index,
                    "page_number": section_chunks[0].get("page_number", 1),
                    "section": section,
                    "chunk_type": "parent",
                }
            )
            parent_index += 1
        for child in section_chunks:
            out.append({**child, "chunk_type": "child"})
    return out


def _injury_criterion_anchor_chunks(
    source: str, pages: list[dict[str, Any]], start_index: int
) -> list[dict[str, Any]]:
    """Small anchor chunks for numbered 'shall not exceed' injury criteria (eval_01 ThCC gap)."""
    anchors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _INJURY_LIMIT_CLAUSE.finditer(source):
        clause = match.group(1).strip()
        body = match.group(2).strip()
        key = f"{clause}|{body[:80]}"
        if key in seen:
            continue
        seen.add(key)
        anchor_text = f"{clause}\n{body}"
        page_number = _page_for_chunk(pages, anchor_text)
        anchors.append(
            {
                "chunk_text": anchor_text,
                "chunk_index": start_index + len(anchors),
                "page_number": page_number,
                "section": clause.rstrip("."),
                "chunk_type": "injury_criterion",
            }
        )
    return anchors


def adaptive_chunk_document(
    path: Path,
    *,
    chunk_size: int = 600,
    overlap: int = 50,
) -> tuple[list[dict[str, Any]], str]:
    """Return chunk dicts and the name of the winning splitter."""
    from core.text_cleanup import strip_unece_boilerplate

    pages = load_pages(path)
    # Phase 1: strip recurring ECE headers/footers before embed + generation.
    pages = [
        {**p, "text": strip_unece_boilerplate(p.get("text") or "")}
        for p in pages
    ]
    source = "\n\n".join(p["text"] for p in pages if p.get("text"))
    if not source.strip():
        raise ValueError(f"No extractable text in {path}")

    candidates = [
        ("recursive_600", _recursive_split(source, 600, overlap)),
        ("recursive_1100", _recursive_split(source, 1100, overlap)),
        ("page", _page_split(pages, chunk_size, overlap)),
    ]
    best_name, best_chunks = max(candidates, key=lambda item: _mean_score(item[1], source))
    logger.info(
        "Adaptive chunking for {}: chose {} ({} chunks, score={:.3f})",
        path.name,
        best_name,
        len(best_chunks),
        _mean_score(best_chunks, source),
    )

    offset = 0
    out: list[dict[str, Any]] = []
    for idx, chunk_text in enumerate(best_chunks):
        page_number = _page_for_chunk(pages, chunk_text)
        section = _detect_section(chunk_text)
        out.append(
            {
                "chunk_text": chunk_text,
                "chunk_index": idx,
                "page_number": page_number,
                "section": section,
                "chunk_type": "adaptive",
            }
        )
        offset += len(chunk_text)

    anchors = _injury_criterion_anchor_chunks(source, pages, start_index=len(out))
    if anchors:
        logger.info("Added {} injury-criterion anchor chunk(s) for {}", len(anchors), path.name)
        out.extend(anchors)

    out = _parent_child_chunks(out)
    return out, best_name
