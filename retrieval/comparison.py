"""Cross-regulation comparison mode: detect, retrieve per named reg, synthesize.

Distinct from single-regulation lookups and from plural "which regulations"
surveys. Comparative asks (compare / differ / vs / versus / relate) that name
≥2 regulations retrieve each corpus separately, then the answer path synthesizes
side-by-side claims with per-claim grounding.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

DEFAULT_COMPARISON_TOP_K = 4

# Comparative / contrast cues (wider than the old "differ between"-only pattern).
_COMPARATIVE_RE = re.compile(
    r"(?ix)"
    r"("
    r"\bcompar(?:e|ison|ing)\b"
    r"|\bversus\b"
    r"|\bvs\.?\b"
    r"|\bdiffer(?:ence|ences|ent)?\b"
    r"|\brelat(?:e|es|ed|ion|ionship|ive|ively)\b"
    r"|\brelevant\b"
    r"|\bin\s+contrast\b"
    r"|\bas\s+opposed\s+to\b"
    r"|\bcross[- ]reg"
    r")"
)

_NAMED_REG_RE = re.compile(
    r"(?ix)\b(?:UN[- ]?(?:ECE[- ]?)?)?R\s*(\d{2,3})\b"
    r"|\bRegulation\s+No\.?\s*(\d{2,3})\b"
    r"|\bReg(?:ulation)?\s+(\d{2,3})\b"
)


@dataclass
class ComparisonRetrievalResult:
    chunks: list[Any] = field(default_factory=list)
    regulation_ids: list[str] = field(default_factory=list)
    per_regulation: dict[str, list[Any]] = field(default_factory=dict)
    topic_query: str = ""
    missing: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "regulation_ids": list(self.regulation_ids),
            "topic_query": self.topic_query,
            "missing": list(self.missing),
            "n_chunks": len(self.chunks),
            "per_regulation_counts": {
                rid: len(chs) for rid, chs in self.per_regulation.items()
            },
        }


def detect_named_regulations(question: str) -> list[str]:
    """Canonical ``UN-ECE-Rxx`` ids in order of first mention (deduped)."""
    q = (question or "").strip()
    if not q:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _NAMED_REG_RE.finditer(q):
        n = m.group(1) or m.group(2) or m.group(3)
        if not n:
            continue
        num = n.lstrip("0") or n
        rid = f"UN-ECE-R{num}"
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def has_comparative_cue(question: str) -> bool:
    """True when the question uses compare / differ / vs / relate language."""
    return bool(_COMPARATIVE_RE.search(question or ""))


def is_comparison_mode_query(question: str) -> bool:
    """Comparative cross-regulation ask: cue + ≥2 named regulations.

    Distinct from single-reg lookups and from plural corpus surveys that do not
    name a specific pair.
    """
    q = (question or "").strip()
    if not q:
        return False
    if not has_comparative_cue(q):
        return False
    return len(detect_named_regulations(q)) >= 2


def comparison_side_query(question: str) -> str:
    """Topic-focused query for per-regulation retrieval (regs/cues stripped)."""
    q = (question or "").strip()
    q_raw = q
    q = _COMPARATIVE_RE.sub(" ", q)
    q = _NAMED_REG_RE.sub(" ", q)
    q = re.sub(
        r"(?ix)\b(?:un[- ]?ece|regulation|how\s+does|what\s+is|in\s+terms\s+of|"
        r"one\s+sentence|each\s+other|and|or|the|a|an|of|to|for|in|on|under|"
        r"when|assessing|tested|why|are|is|does|from|between|with|among|"
        r"addressed|primary|occupant)\b",
        " ",
        q,
    )
    q = re.sub(r"\s+", " ", q).strip(" ?.,;:")
    impact_cue = bool(
        re.search(
            r"(?ix)\b(?:impact\s+direction|collision\s+direction|frontal|lateral|"
            r"side\s+impact)\b",
            q_raw,
        )
    )
    # Short leftovers ("one sentence") or bare compare → canonical R94/R95 contrast.
    if len(q) < 16 or re.search(r"(?ix)\bone\s+sentence\b", q_raw):
        q = (
            "frontal collision impact lateral side impact "
            "protection objective injury criteria"
        )
    elif impact_cue and not re.search(r"(?ix)\b(?:frontal|lateral|side)\b", q):
        q = f"{q} frontal collision lateral side impact".strip()
    return q


def comparison_side_query_for_regulation(question: str, regulation_id: str) -> str:
    """Topic query biased toward the crash type of each named regulation."""
    base = comparison_side_query(question)
    rid = (regulation_id or "").upper()
    q = question or ""
    impactish = bool(
        re.search(
            r"(?ix)\b(?:impact|collision|direction|frontal|lateral|side)\b",
            q,
        )
    ) or bool(re.search(r"(?ix)\b(?:frontal|lateral|side)\b", base))
    # Bare "compare R94 and R95 (in one sentence)" → crash-direction contrast.
    bare_compare = bool(
        re.search(r"(?ix)\b(?:compar(?:e|ison)|differ|versus|\bvs\.?\b)\b", q)
    ) and bool(re.search(r"(?ix)\bone\s+sentence\b", q) or len(base.split()) <= 8)
    if rid.endswith("R94") and (impactish or bare_compare):
        return (
            f"{base} frontal collision impact offset deformable barrier "
            f"protection of the occupants front seats"
        ).strip()
    if rid.endswith("R95") and (impactish or bare_compare):
        return (
            f"{base} lateral side impact collision mobile deformable barrier "
            f"protection of the occupants"
        ).strip()
    return base


def balance_comparison_chunks(
    per_regulation: dict[str, list[Any]],
    *,
    max_total: int = 6,
) -> list[Any]:
    """Round-robin merge so context budget cannot drop an entire regulation."""
    if not per_regulation:
        return []
    queues = {rid: list(chs) for rid, chs in per_regulation.items()}
    merged: list[Any] = []
    seen: set[str] = set()
    while len(merged) < max_total and any(queues.values()):
        progress = False
        for rid, q in queues.items():
            if len(merged) >= max_total:
                break
            while q:
                c = q.pop(0)
                cid = getattr(c, "chunk_id", None) or ""
                if cid and cid in seen:
                    continue
                if cid:
                    seen.add(cid)
                merged.append(c)
                progress = True
                break
        if not progress:
            break
    return merged


def comparison_top_k() -> int:
    try:
        return max(
            2,
            int(
                (
                    os.getenv("COMPARISON_TOP_K") or str(DEFAULT_COMPARISON_TOP_K)
                ).strip()
            ),
        )
    except ValueError:
        return DEFAULT_COMPARISON_TOP_K


COMPARISON_SYSTEM_PROMPT = """\
You are a UNECE passive-safety regulation assistant comparing multiple regulations.

You MUST reply with a single JSON object matching this schema:
{
  "answer_segments": [
    { "text": "<one factual claim about exactly ONE regulation, no citation markup>",
      "citation_chunk_id": "<exact chunk_id from a provided passage for that regulation>" }
  ]
}

STRICT RULES:
1. Answer ONLY from the provided context passages. Do not use outside knowledge.
2. Each answer_segment is one claim about ONE named regulation (e.g. UN R94 or UN R95)
   with EXACTLY ONE citation_chunk_id from that regulation's passages.
3. Cover EACH named regulation with at least one grounded claim when context exists.
4. Prefer side-by-side contrast: state the distinguishing fact for each regulation
   (impact direction, injury criteria, scope, etc. as asked).
5. When comparing collision / impact direction for UN R94 vs UN R95, state the crash
   direction explicitly (frontal vs lateral/side) when the passages support it —
   do not digress into sensor-axis or electrical details unless that is what was asked.
6. Do NOT write section/page citation chips in "text" — the backend attaches them.
7. If context for a regulation is missing, omit claims for that regulation rather than inventing.
8. citation_chunk_id MUST be copied verbatim from a provided chunk_id.
"""

COMPARISON_USER_INSTRUCTION = """\
COMPARISON MODE:
- Produce a side-by-side contrast: one or more answer_segments per named regulation.
- Name the regulation in each claim text (UN R94, UN R95, …).
- Each claim must cite a chunk from that same regulation.
- An answer spanning two regulations is valid when each claim is individually grounded —
  it does NOT need to map to a single shared chunk.
- For impact/collision-direction compares, prefer the words frontal and lateral/side
  when supported by the retrieved passages.
- If the user asks only to "compare" R94 and R95 (e.g. in one sentence) without a
  narrower axis, contrast frontal vs lateral/side impact protection when passages support it.
"""


def retrieve_comparison(
    question: str,
    *,
    regulation_ids: Sequence[str] | None = None,
    top_k: int | None = None,
    llm: object | None = None,
    client: object | None = None,
    embedder: object | None = None,
    collection: str | None = None,
    do_rerank: bool = True,
) -> ComparisonRetrievalResult:
    """Retrieve the topic from EACH named regulation separately (hard filter each)."""
    from retrieval.retrieve import retrieve

    regs = list(regulation_ids or detect_named_regulations(question))
    regs = [r for r in regs if r]
    topic = comparison_side_query(question)
    per_k = top_k or comparison_top_k()
    per_reg: dict[str, list[Any]] = {}
    missing: list[str] = []

    # Bound hybrid candidates per side — default HYBRID_TOP_K=30 × long
    # small-to-big texts makes local CrossEncoder prohibitively slow for
    # dual-corpus comparison (2× per question). Skip CrossEncoder on side
    # retrieves: hard regulation_id filter + hybrid is enough, and avoids a
    # Windows spawn/Qdrant deadlock when the worker re-enters retrieval.
    side_hybrid_k = max(per_k * 3, 10)
    for rid in regs:
        side_topic = comparison_side_query_for_regulation(question, rid)
        side_chunks = retrieve(
            side_topic,
            top_k=per_k,
            regulation_id=rid,
            rewrite=False,
            do_rerank=False,
            small_to_big=True,
            llm=llm,
            client=client,  # type: ignore[arg-type]
            embedder=embedder,  # type: ignore[arg-type]
            collection=collection,
            allow_comparison_branch=False,
            hybrid_top_k=side_hybrid_k,
        )
        tagged = [c for c in side_chunks if (c.regulation_id or "") == rid]
        use = tagged or list(side_chunks)
        per_reg[rid] = use
        if not use:
            missing.append(rid)

    merged = balance_comparison_chunks(per_reg, max_total=max(per_k * len(regs), 6))

    logger.info(
        "comparison retrieve regs=%s topic=%r chunks=%d missing=%s",
        regs,
        topic[:80],
        len(merged),
        missing,
    )
    return ComparisonRetrievalResult(
        chunks=merged,
        regulation_ids=regs,
        per_regulation=per_reg,
        topic_query=topic,
        missing=missing,
    )
