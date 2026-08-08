"""Plural / all-regulation survey retrieval.

When a query uses plural regulation-scope language ("which regulations",
"across all regulations", "in general" without a named reg), run retrieval
separately for each indexed ``regulation_id``, keep only topic-relevant hits,
and expose covered vs missing regs for answer synthesis.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_PER_REGULATION_TOP_K = 3

# Explicit plural / corpus-wide cues.
_PLURAL_SCOPE_RE = re.compile(
    r"(?ix)"
    r"("
    r"\bwhich\s+regulations?\b"
    r"|\bwhat\s+regulations?\b"
    r"|\bwhich\s+of\s+(?:the\s+)?regulations?\b"
    r"|\bacross\s+(?:all\s+)?(?:the\s+)?regulations?\b"
    r"|\bamong\s+(?:all\s+)?(?:the\s+)?regulations?\b"
    r"|\bin\s+(?:all\s+)?(?:the\s+)?(?:indexed\s+)?regulations?\b"
    r"|\ball\s+(?:indexed\s+)?regulations?\b"
    r"|\bfor\s+each\s+regulation\b"
    r"|\bevery\s+regulation\b"
    r")"
)

_IN_GENERAL_RE = re.compile(r"(?i)\bin\s+general\b")
_IN_GENERAL_TOPIC_RE = re.compile(
    r"(?i)\b(requirement|requirements|safety|regulation|regulations|"
    r"impact|vehicle|limit|criterion|criteria|protection)\b"
)

# Boilerplate stripped before topic-term overlap checks.
_TOPIC_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "which",
        "what",
        "where",
        "when",
        "how",
        "who",
        "whom",
        "whose",
        "this",
        "that",
        "these",
        "those",
        "any",
        "all",
        "each",
        "every",
        "across",
        "among",
        "between",
        "under",
        "about",
        "into",
        "over",
        "general",
        "include",
        "includes",
        "including",
        "cover",
        "covers",
        "covered",
        "address",
        "addresses",
        "addressed",
        "have",
        "has",
        "had",
        "there",
        "their",
        "regulation",
        "regulations",
        "un",
        "ece",
        "unece",
        "indexed",
        "please",
        "tell",
        "me",
        "list",
    }
)

# Stem / synonym expansions so "electrical" matches "electric" / "high voltage".
_TERM_EXPAND: dict[str, tuple[str, ...]] = {
    "electrical": ("electrical", "electric", "high voltage", "high-voltage", "reess"),
    "electric": ("electrical", "electric", "high voltage", "high-voltage", "reess"),
    "safety": ("safety", "protection", "safe"),
    "isolation": ("isolation", "isolat"),
    "resistance": ("resistance", "resist"),
}

# When these topic seeds appear in the question, require a domain phrase — a lone
# incidental "electrical" (e.g. "mechanical, electrical, digital tools") is not enough.
_DOMAIN_PHRASE_RE: dict[str, re.Pattern[str]] = {
    "electrical": re.compile(
        r"(?ix)"
        r"("
        r"electrical\s+safety"
        r"|electric\s+safety"
        r"|protection\s+against\s+electrical"
        r"|electrical\s+shock"
        r"|high[\s-]?voltage"
        r"|electrical\s+power\s+train"
        r"|electric\s+power\s+train"
        r"|\breess\b"
        r"|isolation\s+resistance"
        r"|electrolyte\s+(?:spillage|leakage)"
        r"|electric\s+safety\s+assessment"
        r")"
    ),
    "electric": re.compile(
        r"(?ix)"
        r"("
        r"electrical\s+safety"
        r"|electric\s+safety"
        r"|protection\s+against\s+electrical"
        r"|electrical\s+shock"
        r"|high[\s-]?voltage"
        r"|electrical\s+power\s+train"
        r"|electric\s+power\s+train"
        r"|\breess\b"
        r"|isolation\s+resistance"
        r"|electrolyte\s+(?:spillage|leakage)"
        r"|electric\s+safety\s+assessment"
        r")"
    ),
}


@dataclass
class RegulationCoverage:
    regulation_id: str
    relevant: bool
    chunks: list = field(default_factory=list)


@dataclass
class MultiRegulationResult:
    chunks: list
    covered: list[str]
    missing: list[str]
    per_regulation: list[RegulationCoverage] = field(default_factory=list)


def is_plural_regulation_query(question: str) -> bool:
    """True for corpus-wide / plural regulation-scope asks."""
    q = (question or "").strip()
    if not q:
        return False
    if _PLURAL_SCOPE_RE.search(q):
        return True
    # "in general" only when no single regulation is named and a topic word remains.
    if _IN_GENERAL_RE.search(q) and _IN_GENERAL_TOPIC_RE.search(q):
        from retrieval.enumerative import detect_named_regulation

        if detect_named_regulation(q) is None:
            return True
    return False


def per_regulation_top_k() -> int:
    try:
        return max(
            1,
            int(
                (
                    os.getenv("MULTI_REG_TOP_K") or str(DEFAULT_PER_REGULATION_TOP_K)
                ).strip()
            ),
        )
    except ValueError:
        return DEFAULT_PER_REGULATION_TOP_K


def topic_terms(question: str) -> list[str]:
    """Content tokens used to decide whether a chunk is on-topic."""
    q = (question or "").lower()
    q = re.sub(r"[^a-z0-9\s/+-]", " ", q)
    out: list[str] = []
    seen: set[str] = set()
    for tok in q.split():
        if len(tok) < 3 or tok in _TOPIC_STOP:
            continue
        if tok.isdigit():
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _term_variants(term: str) -> tuple[str, ...]:
    return _TERM_EXPAND.get(term, (term,))


def chunk_matches_topic(chunk: object, terms: Sequence[str]) -> bool:
    """True when chunk text overlaps enough topic terms (or synonym expansions)."""
    if not terms:
        return bool(getattr(chunk, "text", None))
    blob = " ".join(
        [
            str(getattr(chunk, "text", "") or ""),
            str(getattr(chunk, "section_title", "") or ""),
            str(getattr(chunk, "section_number", "") or ""),
        ]
    ).lower()
    if not blob.strip():
        return False

    # Domain seeds (electrical/electric): require a real EV/HV safety phrase.
    for seed, pat in _DOMAIN_PHRASE_RE.items():
        if seed in terms and not pat.search(blob):
            return False

    def _hit(term: str) -> bool:
        if term in _DOMAIN_PHRASE_RE:
            # Already validated via domain phrase above.
            return True
        for v in _term_variants(term):
            v = (v or "").lower().strip()
            if not v:
                continue
            if " " in v:
                if v in blob:
                    return True
            elif re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", blob):
                return True
        return False

    hits = sum(1 for term in terms if _hit(term))
    # Drop ultra-generic survey nouns from the quota ("requirements").
    scored_terms = [t for t in terms if t not in {"requirements", "requirement", "provisions", "provision"}]
    if not scored_terms:
        scored_terms = list(terms)
    scored_hits = sum(1 for term in scored_terms if _hit(term))
    if len(scored_terms) <= 3:
        need = len(scored_terms)
    else:
        need = max(2, (len(scored_terms) + 1) // 2)
    return scored_hits >= need and hits >= 1
def regulation_has_relevant_content(
    chunks: Sequence[object],
    question: str,
) -> bool:
    terms = topic_terms(question)
    return any(chunk_matches_topic(c, terms) for c in chunks)


def short_regulation_label(regulation_id: str) -> str:
    rid = (regulation_id or "").strip()
    if rid.startswith("UN-ECE-"):
        return "UN " + rid[len("UN-ECE-") :]
    return rid or "?"


def format_coverage_summary(*, covered: Sequence[str], missing: Sequence[str]) -> str:
    """Deterministic covered/missing lines (always appended for survey answers)."""
    lines: list[str] = []
    if covered:
        labels = [short_regulation_label(r) for r in covered]
        lines.append(
            "Relevant content on this topic was found in: " + ", ".join(labels) + "."
        )
    else:
        lines.append(
            "No relevant content on this topic was found in any indexed regulation."
        )
    if missing:
        labels = [short_regulation_label(r) for r in missing]
        lines.append(
            "No relevant content on this topic was found in the following "
            "indexed regulations: " + ", ".join(labels) + "."
        )
    return "\n".join(lines)


def retrieve_per_indexed_regulation(
    query: str,
    *,
    top_k: int | None = None,
    llm: object | None = None,
    rewrite: bool = False,
    rewrite_result: object | None = None,
    do_rerank: bool = True,
    small_to_big: bool = False,
    client: object | None = None,
    embedder: object | None = None,
    collection: str | None = None,
) -> MultiRegulationResult:
    """Run ``retrieve`` once per indexed regulation; keep topic-relevant hits only."""
    from retrieval.retrieve import get_indexed_regulations, retrieve

    per_k = top_k or per_regulation_top_k()
    regs = [
        r.regulation_id
        for r in get_indexed_regulations(
            client=client,  # type: ignore[arg-type]
            collection=collection,
        )
    ]
    regs = sorted({r for r in regs if r})
    if not regs:
        logger.warning("multi-regulation retrieve: no indexed regulations")
        return MultiRegulationResult(chunks=[], covered=[], missing=[])

    terms = topic_terms(query)
    covered: list[str] = []
    missing: list[str] = []
    merged: list = []
    per_rows: list[RegulationCoverage] = []
    seen_ids: set[str] = set()

    for rid in regs:
        hits = retrieve(
            query,
            top_k=per_k,
            regulation_id=rid,
            llm=llm,
            rewrite=rewrite,
            rewrite_result=rewrite_result,
            do_rerank=do_rerank,
            small_to_big=small_to_big,
            client=client,  # type: ignore[arg-type]
            embedder=embedder,  # type: ignore[arg-type]
            collection=collection,
            history=None,
        )
        relevant_hits = [c for c in hits if chunk_matches_topic(c, terms)]
        is_rel = bool(relevant_hits)
        row = RegulationCoverage(
            regulation_id=rid,
            relevant=is_rel,
            chunks=relevant_hits[:per_k] if is_rel else [],
        )
        per_rows.append(row)
        if is_rel:
            covered.append(rid)
            for c in row.chunks:
                cid = getattr(c, "chunk_id", "") or ""
                if cid and cid in seen_ids:
                    continue
                if cid:
                    seen_ids.add(cid)
                merged.append(c)
        else:
            missing.append(rid)
        logger.info(
            "multi-regulation retrieve rid=%s relevant=%s n_hits=%d n_kept=%d",
            rid,
            is_rel,
            len(hits),
            len(row.chunks),
        )

    logger.info(
        "multi-regulation done covered=%s missing=%s n_chunks=%d terms=%s",
        covered,
        missing,
        len(merged),
        terms,
    )
    return MultiRegulationResult(
        chunks=merged,
        covered=covered,
        missing=missing,
        per_regulation=per_rows,
    )


def record_multi_regulation_on_trace(result: MultiRegulationResult) -> None:
    """Attach coverage metadata to the current query trace (if any)."""
    try:
        from observability.context import get_current_trace

        tr = get_current_trace()
        if tr is None:
            return
        tr.optimizations["multi_regulation"] = True
        tr.optimizations["multi_regulation_covered"] = list(result.covered)
        tr.optimizations["multi_regulation_missing"] = list(result.missing)
        tr.optimizations["multi_regulation_n_chunks"] = len(result.chunks)
    except Exception:  # noqa: BLE001
        pass
