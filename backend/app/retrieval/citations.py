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
import os
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
    reg_code = doc.get("regulation") or doc.get("doc_id") or ""
    meta: DocumentMeta = get_document_meta(reg_code)
    heading = doc.get("heading_path", "") or ""
    page = parse_page(heading)
    section = parse_section(
        heading,
        doc.get("title", "") or doc.get("section_title", "") or "",
    )
    clause_num = doc.get("clause_number") or doc.get("clause")
    if clause_num and section and str(clause_num) not in str(section):
        section = str(clause_num)

    # Human-readable locator: prefer clause/section; include page when known.
    locator_parts: list[str] = []
    if section:
        locator_parts.append(f"§{section}")
    if page is not None:
        locator_parts.append(f"p.{page}")
    locator = ", ".join(locator_parts) or (heading or "n/a")

    chunk_rev = doc.get("revision")
    if chunk_rev and str(chunk_rev).lower() not in ("unknown", "n/a", "synthetic"):
        rev = str(chunk_rev)
        revision_verified = meta.verified
    else:
        rev = meta.indexed_revision or "revision unverified"
        revision_verified = meta.verified
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
        "revision": meta.indexed_revision or chunk_rev,
        "revision_verified": revision_verified,
        "legal_reference": meta.legal_reference,
        "page": page,
        "section": section,
        "heading_path": heading,
        "snippet": _snippet(doc.get("text", "")),
        "chunk_id": doc.get("id"),
        "rerank_score": doc.get("rerank_score"),
        "semantic_score": doc.get("semantic_score"),
        "source": doc.get("source"),
        "is_synthetic": bool(doc.get("is_synthetic")),
    }


# build_citations defined after enrich_doc_provenance (below)


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def confidence_band(confidence: float) -> str:
    """Map grounding confidence to high / medium / low bands."""
    high = float(os.getenv("CONFIDENCE_BAND_HIGH", "0.65"))
    medium = float(os.getenv("CONFIDENCE_BAND_MEDIUM", "0.45"))
    if confidence >= high:
        return "high"
    if confidence >= medium:
        return "medium"
    return "low"


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
            "confidence_band": "low",
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
        "confidence_band": confidence_band(confidence),
        "reason": reason,
        "best_semantic": round(best_sem, 4) if best_sem is not None else None,
        "best_rerank_prob": (
            round(best_rerank_prob, 4) if best_rerank_prob is not None else None
        ),
    }


# Markers injected into the context look like [S1], [S2] …; the LLM is required
# to cite them inline. We use this to tie answer-level flags to the sources the
# answer ACTUALLY used, not to every retrieved passage.
_MARKER_RE = re.compile(r"\[S(\d+)\]")
_CHUNK_HEADER_RE = re.compile(
    r"^\[([A-Z0-9_]+)\s*\|\s*([^|\]]+)\s*\|\s*([^\]]+)\]",
    re.I,
)
_APPLICABILITY_BLOCK_RE = re.compile(
    r"^APPLICABILITY:.*?\n---\n",
    re.S | re.I,
)


def _parse_chunk_header(text: str) -> dict[str, str]:
    """Extract doc_id / revision / clause from chunker prepend header."""
    m = _CHUNK_HEADER_RE.search(text or "")
    if not m:
        return {}
    return {
        "regulation": m.group(1).strip().upper(),
        "revision": m.group(2).strip(),
        "clause": m.group(3).strip(),
    }


def enrich_doc_provenance(
    doc: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Merge chunk-index metadata and parent provenance so citations never show
    as 'Unknown document' when only the APPLICABILITY sub-chunk was retrieved.
    """
    enriched = dict(doc)
    cid = doc.get("id") or doc.get("chunk_id")
    chunk = (chunk_lookup or {}).get(cid, {}) if cid else {}

    for key in (
        "regulation",
        "doc_id",
        "heading_path",
        "section_title",
        "clause_number",
        "clause",
        "revision",
        "doc_type",
        "parent_id",
        "title",
    ):
        if not enriched.get(key) and chunk.get(key):
            enriched[key] = chunk[key]

    header = _parse_chunk_header(enriched.get("text", "") or chunk.get("text", ""))
    if not enriched.get("regulation") and header.get("regulation"):
        enriched["regulation"] = header["regulation"]
    if not enriched.get("revision") and header.get("revision"):
        rev = header["revision"]
        if rev.lower() not in ("unknown", "n/a"):
            enriched["revision"] = rev
    if not enriched.get("clause_number") and header.get("clause"):
        enriched["clause_number"] = header["clause"]

    parent_id = enriched.get("parent_id") or chunk.get("parent_id")
    if parent_id and chunk_lookup and parent_id in chunk_lookup:
        parent = chunk_lookup[parent_id]
        for key in ("regulation", "doc_id", "revision", "doc_type", "heading_path"):
            if not enriched.get(key) and parent.get(key):
                enriched[key] = parent[key]
        if not enriched.get("section_title") and parent.get("section_title"):
            enriched["section_title"] = parent["section_title"]

    related_to = enriched.get("related_to")
    if related_to and chunk_lookup and related_to in chunk_lookup:
        src = chunk_lookup[related_to]
        for key in ("regulation", "doc_id", "revision", "doc_type"):
            if not enriched.get(key) and src.get(key):
                enriched[key] = src[key]

    return enriched


def build_citations(
    documents: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return [
        build_citation(enrich_doc_provenance(d, chunk_lookup), i + 1)
        for i, d in enumerate(documents)
    ]


def extract_cited_markers(answer_text: str) -> set[str]:
    """Parse the source markers ([S1], [S2] …) actually cited in the answer."""
    if not answer_text:
        return set()
    return {f"S{m.group(1)}" for m in _MARKER_RE.finditer(answer_text)}


def derive_answer_flags(
    citations: list[dict[str, Any]],
    answer_text: str | None = None,
    *,
    should_abstain: bool = False,
) -> list[dict[str, Any]]:
    """
    Answer-level flags:
      - multiple revisions exist for a cited legal regulation -> confirm version
      - both legal regulations and rating protocols cited -> do not blur them
      - a cited revision is unverified

    Flags are scoped to the sources the answer actually cited. When `answer_text`
    is provided, only citations whose marker ([S#]) appears in the answer are
    considered; this prevents a revision warning from firing for a regulation
    that was merely retrieved but never used. Flags are deduplicated by
    (type, regulation). When the system abstains there is no answer, so no flags.
    """
    if should_abstain:
        return []

    # Restrict to the citations the answer actually used (if we know the answer).
    if answer_text is not None:
        cited_markers = extract_cited_markers(answer_text)
        citations = [c for c in citations if c.get("marker") in cited_markers]

    if not citations:
        return []

    flags: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()  # (flag_type, regulation_code) -> dedupe

    cited_regs = {c["regulation"] for c in citations if c.get("regulation")}
    for reg in sorted(cited_regs):
        meta = get_document_meta(reg)
        if meta.is_legal and meta.has_multiple_revisions:
            key = ("multiple_revisions", meta.code)
            if key not in seen:
                seen.add(key)
                flags.append({
                    "type": "multiple_revisions",
                    "regulation": meta.display_name,
                    "message": (
                        f"{meta.display_name} has multiple revisions. This answer "
                        f"is based on {meta.indexed_revision}. Confirm this matches "
                        f"the revision applicable to your project."
                    ),
                })
        if not meta.verified and meta.code != "UNKNOWN":
            key = ("unverified_revision", meta.code)
            if key not in seen:
                seen.add(key)
                flags.append({
                    "type": "unverified_revision",
                    "regulation": meta.display_name,
                    "message": (
                        f"The indexed revision of {meta.display_name} is not "
                        f"verified; treat version-specific values with caution."
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


def validate_citation_attribution(citation: dict[str, Any]) -> list[str]:
    """Return failure codes when a citation lacks real provenance."""
    failures: list[str] = []
    reg = (citation.get("regulation") or "").upper()
    if not reg or reg == "UNKNOWN":
        failures.append("missing_regulation")
    if citation.get("document") == "Unknown document":
        failures.append("unknown_document")
    rev = str(citation.get("revision") or "")
    if not rev or rev.lower() == "revision unverified":
        failures.append("unverified_revision")
    label = str(citation.get("doc_type_label") or "")
    if not label or label.lower() == "unknown":
        failures.append("invalid_doc_type")
    if "unverified" in str(citation.get("label") or "").lower():
        failures.append("unverified_label")
    return failures


def detect_query_vehicle_categories(query: str) -> set[str]:
    from backend.app.retrieval.applicability_boost import detect_query_categories

    return detect_query_categories(query)


def chunk_authority_for_categories(
    doc: dict[str, Any],
    query_categories: set[str],
) -> bool:
    """
    True when a chunk's applicability header supports citing it as authority for
    the requested vehicle category — not merely because the category is mentioned
    in contrast text.
    """
    if not query_categories:
        return True
    applies = doc.get("applies_to_category") or []
    if isinstance(applies, str):
        applies = [applies]
    if not applies:
        return True

    app_set = set(applies)
    if "M1_N1" in query_categories:
        if "M1_N1" in app_set:
            return True
        if "NOT_M1_N1" in app_set and "M1_N1" not in app_set:
            return False
    if "M3_N3" in query_categories:
        return "M3_N3" in app_set or ("NOT_M1_N1" in app_set and "M1_N1" not in app_set)
    if "M2_N2" in query_categories:
        return "M2_N2" in app_set or "NOT_M1_N1" in app_set
    if "NOT_M1_N1" in query_categories:
        return "NOT_M1_N1" in app_set or bool(app_set - {"M1_N1"})
    return bool(app_set.intersection(query_categories))


def validate_category_citation_authority(
    query: str,
    citations: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return flags when cited sources lack applicability authority for query category."""
    q_cats = detect_query_vehicle_categories(query)
    if not q_cats:
        return []

    flags: list[dict[str, Any]] = []
    for cite, doc in zip(citations, documents):
        enriched = enrich_doc_provenance(doc, chunk_lookup)
        if not chunk_authority_for_categories(enriched, q_cats):
            flags.append({
                "type": "applicability_mismatch",
                "regulation": cite.get("document"),
                "message": (
                    f"Citation [{cite.get('marker')}] applicability "
                    f"({enriched.get('applies_to_category')}) may not be authoritative "
                    f"for the vehicle category in this query — verify clause match."
                ),
            })
    return flags


_OUTSIDE_DISCLOSURE_RE = re.compile(
    r"general knowledge\s*\(outside retrieved corpus\)|outside retrieved corpus|"
    r"\[outside corpus\]|outside-corpus knowledge|not from (?:the )?retrieved",
    re.I,
)
_CLASSIFICATION_FACT_RE = re.compile(
    r"(?:≤|<=)\s*8\s*seats|8\s*passenger\s*seats|excluding\s+(?:the\s+)?driver|"
    r"M1\s*(?:=|means|refers to|is\s+(?:a|the))|passenger car(?:s)?\s+(?:with|having)|"
    r"vehicle\s+category\s+M1\s+(?:means|is|refers)",
    re.I,
)
_SCOPE_NARROW_RE = re.compile(
    r"\b(M1|N1|M2|M3|N2|N3|side[- ]facing|rearward[- ]facing|outboard|"
    r"front\s+centre|upper\s+deck)\b",
    re.I,
)
_BROAD_HEADLINE_RE = re.compile(
    r"(?:^|\n)\s*(?:UN\s+R\d+|the\s+regulation)\s+(?:addresses|covers|requires|"
    r"specifies)\s+(?:seat|anchorage|crash|structural)",
    re.I,
)


def detect_knowledge_boundary_flags(
    query: str,
    answer: str,
    *,
    should_abstain: bool = False,
) -> list[dict[str, Any]]:
    """Flag undisclosed outside-corpus knowledge (classification facts, etc.)."""
    if should_abstain or not answer:
        return []
    low = answer.lower()
    if any(m in low for m in ("not found in the regulations", "insufficient data", "not in the corpus")):
        return []

    flags: list[dict[str, Any]] = []
    has_outside_fact = bool(_CLASSIFICATION_FACT_RE.search(answer))
    disclosed = bool(_OUTSIDE_DISCLOSURE_RE.search(answer))

    if has_outside_fact and not disclosed:
        flags.append({
            "type": "outside_knowledge_undisclosed",
            "message": (
                "Answer appears to use general/outside-corpus knowledge (e.g. vehicle "
                "classification) without the required 'General knowledge (outside "
                "retrieved corpus):' label."
            ),
        })
    elif has_outside_fact and disclosed:
        flags.append({
            "type": "outside_knowledge_disclosed",
            "message": "Outside-corpus knowledge is explicitly labeled (expected for interpretive facts).",
        })
    return flags


_CORPUS_UNCERTAINTY_DENIAL_RE = re.compile(
    r"(?:not (?:found|present|included|in)|(?:does not|doesn't) (?:appear|exist|contain)|"
    r"no (?:evidence|mention|requirement)|is not (?:in|addressed|covered)|"
    r"cannot find|not in (?:the )?(?:regulation|corpus|retrieved))",
    re.I,
)
_CORPUS_UNCERTAINTY_DISCLOSED_RE = re.compile(
    r"not confident|cannot verify in (?:the )?corpus|uncertain whether|"
    r"may appear in a section not retrieved|based on (?:the )?retrieved passages",
    re.I,
)
_PILOT_CORPUS_RE = re.compile(r"UN\s*R(?:14|16)|regulation\s+(?:14|16)", re.I)


def detect_corpus_uncertainty_flags(
    query: str,
    answer: str,
    *,
    should_abstain: bool = False,
) -> list[dict[str, Any]]:
    """Flag definitive corpus denials without disclosed uncertainty (Part F)."""
    if should_abstain or not answer:
        return []
    if not _PILOT_CORPUS_RE.search(query):
        return []
    if not _CORPUS_UNCERTAINTY_DENIAL_RE.search(answer):
        return []
    if _CORPUS_UNCERTAINTY_DISCLOSED_RE.search(answer):
        return [{
            "type": "corpus_uncertainty_disclosed",
            "message": (
                "Answer appropriately discloses uncertainty about corpus coverage."
            ),
        }]
    return [{
        "type": "corpus_uncertainty_undisclosed",
        "message": (
            "Answer denies or negates a corpus requirement without the required "
            "'not confident based on retrieved passages' uncertainty disclosure."
        ),
    }]


def detect_scope_overclaim_flags(
    answer: str,
    citations: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Flag when the headline generalizes beyond cited subset-specific evidence."""
    if not answer or not citations:
        return []

    cited_markers = extract_cited_markers(answer)
    if not cited_markers:
        return []

    narrow_tokens: set[str] = set()
    for cite, doc in zip(citations, documents):
        if cite.get("marker") not in cited_markers:
            continue
        enriched = enrich_doc_provenance(doc, chunk_lookup)
        applies = enriched.get("applies_to_category") or []
        if isinstance(applies, str):
            applies = [applies]
        text = (enriched.get("text") or "").lower()
        for cat in applies:
            if cat not in ("ALL", "GENERAL"):
                narrow_tokens.add(cat.replace("_", "/"))
        for tok in ("side-facing", "side facing", "rearward", "outboard", "upper deck"):
            if tok in text:
                narrow_tokens.add(tok)

    if not narrow_tokens:
        return []

    headline = re.split(r"\n\s*\n", answer.strip())[0][:250]
    headline_has_scope = any(
        tok.lower() in headline.lower() for tok in narrow_tokens
    ) or bool(_SCOPE_NARROW_RE.search(headline))

    if headline_has_scope:
        return []

    if _BROAD_HEADLINE_RE.search(headline):
        return [{
            "type": "scope_overclaim",
            "message": (
                "Opening claim may generalize beyond subset-specific cited evidence "
                f"({', '.join(sorted(narrow_tokens)[:4])}). State the narrowest scope "
                "in the headline."
            ),
        }]
    return []


def detect_category_value_misattribution(
    query: str,
    answer: str,
    citations: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Flag when answer assigns a category-specific value from an excluding chunk."""
    q_cats = detect_query_vehicle_categories(query)
    if not q_cats or not answer:
        return []

    flags: list[dict[str, Any]] = []
    cited_markers = extract_cited_markers(answer)

    if "M1_N1" in q_cats:
        if re.search(r"M1[^.\n]{0,60}\b675\b", answer, re.I) and not re.search(
            r"1[,.]?350|1350", answer
        ):
            flags.append({
                "type": "category_value_misattribution",
                "message": "M1/N1 query: 675 daN is typically for categories other than M1/N1 — verify attribution.",
            })

    for cite, doc in zip(citations, documents):
        if cite.get("marker") not in cited_markers:
            continue
        enriched = enrich_doc_provenance(doc, chunk_lookup)
        if not chunk_authority_for_categories(enriched, q_cats):
            flags.append({
                "type": "category_value_misattribution",
                "regulation": cite.get("document"),
                "message": (
                    f"Value cited via [{cite.get('marker')}] may be misattributed — "
                    f"chunk applicability ({enriched.get('applies_to_category')}) "
                    f"does not authorize the queried vehicle category."
                ),
            })
    return flags
