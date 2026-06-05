"""
Citation building and grounding assessment (Item 1: grounding & anti-hallucination).

Every retrieved chunk is turned into a structured citation carrying:
  - document name (display + full title)
  - page and/or section (clause) reference
  - source revision / amendment
  - document type (legal regulation vs rating protocol)

The grounding assessment decides whether retrieval is confident enough to let the
LLM answer at all. If not, the caller must reply "not found in the corpus" rather
than generating an answer.
"""

from __future__ import annotations

import math
import re
from typing import Any

from backend.app.core.document_registry import (
    DocumentMeta,
    doc_type_label,
    get_document_meta,
)

# heading_path looks like "UN_R14 > Page 5" (page-based) or
# "UN_R14 > 2.4 1. > 2.4.1 1." (clause-based).
_PAGE_RE = re.compile(r"page\s+(\d+)", re.I)
_CLAUSE_RE = re.compile(r"\b(\d+(?:\.\d+){0,3})\b")


def parse_page(heading_path: str) -> int | None:
    if not heading_path:
        return None
    m = _PAGE_RE.search(heading_path)
    return int(m.group(1)) if m else None


def parse_section(heading_path: str, section_title: str = "") -> str | None:
    """
    Prefer the most specific clause number in the heading path (e.g. "2.4.1"),
    which is the legally meaningful citation unit. Fall back to the section title.
    """
    if heading_path:
        # Drop the leading regulation token, keep the deepest path segment.
        segments = [s.strip() for s in heading_path.split(">") if s.strip()]
        tail = segments[-1] if segments else ""
        if "page" not in tail.lower():
            m = _CLAUSE_RE.search(tail)
            if m:
                return m.group(1)
            if tail:
                return tail[:60]
    if section_title and "page" not in section_title.lower():
        return section_title[:60]
    return None


def _snippet(text: str, limit: int = 240) -> str:
    # Strip the leading "[heading]" / "# heading" marker the chunker prepends.
    body = re.sub(r"^[#\[].*?\]?\n+", "", (text or ""), count=1).strip()
    body = re.sub(r"\s+", " ", body)
    return body[:limit] + ("…" if len(body) > limit else "")


def build_citation(doc: dict[str, Any], index: int) -> dict[str, Any]:
    """Turn one retrieved doc into a structured, displayable citation."""
    reg_code = doc.get("regulation", "")
    meta: DocumentMeta = get_document_meta(reg_code)
    heading = doc.get("heading_path", "") or ""
    page = parse_page(heading)
    section = parse_section(heading, doc.get("title", ""))

    # Human-readable locator: prefer clause/section; include page when known.
    locator_parts: list[str] = []
    if section:
        locator_parts.append(f"§{section}")
    if page is not None:
        locator_parts.append(f"p.{page}")
    locator = ", ".join(locator_parts) or (heading or "n/a")

    rev = meta.indexed_revision or "revision unverified"
    label = f"{meta.display_name} ({rev}), {locator}"

    return {
        "marker": f"S{index}",
        "label": label,
        "document": meta.display_name,
        "full_title": meta.full_title,
        "regulation": reg_code,
        "doc_type": meta.doc_type,
        "doc_type_label": doc_type_label(meta.doc_type),
        "is_legal": meta.is_legal,
        "authority": meta.authority,
        "revision": meta.indexed_revision,
        "revision_verified": meta.verified,
        "legal_reference": meta.legal_reference,
        "page": page,
        "section": section,
        "heading_path": heading,
        "snippet": _snippet(doc.get("text", "")),
        "chunk_id": doc.get("id"),
        "rerank_score": doc.get("rerank_score"),
        "semantic_score": doc.get("semantic_score"),
        "source": doc.get("source"),
    }


def build_citations(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_citation(d, i + 1) for i, d in enumerate(documents)]


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def assess_grounding(
    documents: list[dict[str, Any]],
    *,
    reranker_used: bool,
    min_semantic: float,
    min_rerank_prob: float,
) -> dict[str, Any]:
    """
    Decide whether retrieval is confident enough to answer.

    Signals (whichever are available):
      - best raw semantic cosine similarity (0..1, interpretable)
      - best cross-encoder rerank score -> sigmoid -> probability (0..1)

    Abstain only when there IS a usable signal and it is below threshold, or when
    no documents were retrieved. If neither signal is available (e.g. BM25-only
    mode with reranker off) we do not block, since we cannot assess confidence.
    """
    if not documents:
        return {
            "should_abstain": True,
            "confidence": 0.0,
            "reason": "no_documents",
            "best_semantic": None,
            "best_rerank_prob": None,
        }

    sem_scores = [
        d["semantic_score"] for d in documents if d.get("semantic_score") is not None
    ]
    best_sem = max(sem_scores) if sem_scores else None

    best_rerank_prob = None
    if reranker_used:
        rr = [
            d["rerank_score"] for d in documents if d.get("rerank_score") is not None
        ]
        if rr:
            best_rerank_prob = _sigmoid(max(rr))

    sem_ok = best_sem is not None and best_sem >= min_semantic
    rerank_ok = best_rerank_prob is not None and best_rerank_prob >= min_rerank_prob
    has_signal = best_sem is not None or best_rerank_prob is not None

    if not has_signal:
        should_abstain = False
        reason = "no_confidence_signal"
    else:
        should_abstain = not (sem_ok or rerank_ok)
        reason = "below_threshold" if should_abstain else "confident"

    confidence = max(
        best_sem or 0.0,
        best_rerank_prob or 0.0,
    )

    return {
        "should_abstain": should_abstain,
        "confidence": round(confidence, 4),
        "reason": reason,
        "best_semantic": round(best_sem, 4) if best_sem is not None else None,
        "best_rerank_prob": (
            round(best_rerank_prob, 4) if best_rerank_prob is not None else None
        ),
    }


def derive_answer_flags(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Answer-level flags required by Item 1:
      - multiple revisions exist for a cited legal regulation -> confirm version
      - both legal regulations and rating protocols cited -> do not blur them
      - a cited revision is unverified
    """
    flags: list[dict[str, Any]] = []
    cited_regs = {c["regulation"] for c in citations if c.get("regulation")}

    for reg in sorted(cited_regs):
        meta = get_document_meta(reg)
        if meta.is_legal and meta.has_multiple_revisions:
            flags.append({
                "type": "multiple_revisions",
                "regulation": meta.display_name,
                "message": (
                    f"{meta.display_name} has multiple revisions. This answer is "
                    f"based on {meta.indexed_revision}. Confirm this matches the "
                    f"revision applicable to your project."
                ),
            })
        if not meta.verified and meta.code != "UNKNOWN":
            flags.append({
                "type": "unverified_revision",
                "regulation": meta.display_name,
                "message": (
                    f"The indexed revision of {meta.display_name} is not verified; "
                    f"treat version-specific values with caution."
                ),
            })

    doc_types = {c["doc_type"] for c in citations}
    if "legal_regulation" in doc_types and "rating_protocol" in doc_types:
        flags.append({
            "type": "mixed_doc_types",
            "message": (
                "This answer references both legal regulations (binding) and "
                "rating protocols (consumer assessment, not legally binding). "
                "These are different in status and must not be treated the same."
            ),
        })
    return flags
