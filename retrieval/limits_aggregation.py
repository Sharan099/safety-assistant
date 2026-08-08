"""Limits / criteria aggregation — structured table answers from Fix 22 data.

Detects \"summarize all (injury) limits / pass-fail criteria\" asks and renders
the verified ``data/limits/<regulation_id>.json`` table as markdown (one row per
criterion, each cited) instead of an LLM prose paragraph.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Aggregation cues — narrower than generic enumerative "list all requirements".
_LIMITS_AGG_RE = re.compile(
    r"(?ix)"
    r"("
    r"\bsummar(?:ize|ise)\s+all\b.{0,80}?\b(?:injury\s+)?(?:limits?|criteria|thresholds?)\b"
    r"|\bsummar(?:ize|ise)\s+all\b.{0,80}?\bpass\s*/\s*fail\b"
    r"|\blist\s+all\b.{0,80}?\b(?:injury\s+)?(?:limits?|criteria|thresholds?)\b"
    r"|\ball\s+(?:the\s+)?(?:frontal\s+)?(?:impact\s+)?(?:injury\s+)?limits?\b"
    r"|\ball\s+(?:the\s+)?(?:pass\s*/\s*fail\s+)?(?:performance\s+)?criteria\b"
    r"|\ball\s+(?:the\s+)?injury\s+criteria\b"
    r")"
)

# Exclude door/topic requirement dumps from the limits-table path.
_NOT_LIMITS_TOPIC_RE = re.compile(
    r"(?ix)\b(?:related\s+to|concerning|regarding)\s+"
    r"(?:doors?|seats?|belts?|reess|pillars?)\b"
)


def is_limits_aggregation_query(question: str) -> bool:
    """True for summarize/list-all limits or pass/fail criteria (table mode)."""
    q = (question or "").strip()
    if not q:
        return False
    if _NOT_LIMITS_TOPIC_RE.search(q):
        return False
    return bool(_LIMITS_AGG_RE.search(q))


def resolve_limits_regulation_id(
    question: str,
    *,
    regulation_id: str | None = None,
    routed: Any | None = None,
) -> str | None:
    if regulation_id:
        return regulation_id
    rid = getattr(routed, "regulation_id", None) if routed is not None else None
    if rid:
        return str(rid)
    try:
        from retrieval.enumerative import (
            detect_named_regulation,
            resolve_hard_regulation_filter,
        )

        named = resolve_hard_regulation_filter(question) or detect_named_regulation(
            question
        )
        if named:
            return named
    except Exception:  # noqa: BLE001
        pass
    q = (question or "").lower()
    # Impact-type cue when the engineer omits the regulation name.
    if re.search(r"\bfrontal\b", q):
        return "UN-ECE-R94"
    if re.search(r"\b(?:side|lateral)\b", q):
        return "UN-ECE-R95"
    return None


def _load_limits_rows(regulation_id: str) -> list[Any]:
    from ingestion.extract_limits import load_limits_table, seed_known_limits

    table = load_limits_table(regulation_id)
    if table is None or not table.limits:
        seed_known_limits()
        table = load_limits_table(regulation_id)
    if table is None:
        return []
    return list(table.limits)


def format_limits_markdown_table(
    limits: Sequence[Any],
    *,
    regulation_id: str,
    chunks_by_id: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Return ``(markdown_table, source_chunk_ids)`` — one row per criterion."""
    chunks_by_id = chunks_by_id or {}
    if not limits:
        return (
            f"_No verified limits table found for {regulation_id}._",
            [],
        )

    header = (
        "| Criterion | Limit | Unit | Section | Citation |\n"
        "|---|---|---|---|---|"
    )
    lines = [header]
    source_ids: list[str] = []
    seen: set[str] = set()

    for row in limits:
        name = (getattr(row, "criterion_name", None) or "criterion").strip()
        aliases = list(getattr(row, "aliases", None) or [])
        short = next(
            (a for a in aliases if re.fullmatch(r"[A-Za-z]{2,8}", str(a) or "")),
            "",
        )
        label = f"{name} ({short})" if short and short.upper() not in name.upper() else name
        op = getattr(row, "operator", "<=") or "<="
        val = getattr(row, "limit_value", None)
        unit = (getattr(row, "unit", None) or "").strip()
        sec = (getattr(row, "section_number", None) or "").strip()
        cid = (getattr(row, "source_chunk_id", None) or "").strip()
        chunk = chunks_by_id.get(cid) if cid else None
        if chunk is None and sec:
            chunk = next(
                (
                    c
                    for c in chunks_by_id.values()
                    if (getattr(c, "section_number", "") or "") == sec
                    and (getattr(c, "regulation_id", "") or "") == regulation_id
                ),
                None,
            )
        limit_s = f"{op} {val}".strip() if val is not None else "—"
        unit_s = unit or "—"
        sec_s = f"§{sec}" if sec else "—"
        if chunk is not None:
            cite = chunk.citation_tag()
            cid2 = (chunk.chunk_id or cid or "").strip()
        elif cid:
            cite = f"`{cid}`"
            cid2 = cid
        elif sec:
            cite = f"[{regulation_id} §{sec}]"
            cid2 = ""
        else:
            cite = "—"
            cid2 = ""
        lines.append(
            f"| {label} | {limit_s} | {unit_s} | {sec_s} | {cite} |"
        )
        if cid2 and cid2 not in seen:
            seen.add(cid2)
            source_ids.append(cid2)

    return "\n".join(lines), source_ids


def render_limits_aggregation_answer(
    question: str,
    *,
    regulation_id: str | None = None,
    chunks: Sequence[Any] | None = None,
    routed: Any | None = None,
    to_source: Any | None = None,
) -> tuple[str, list[Any]]:
    """Build a cited markdown limits table for aggregation asks."""
    rid = resolve_limits_regulation_id(
        question, regulation_id=regulation_id, routed=routed
    )
    if not rid:
        return (
            "Name a regulation (e.g. UN R94) to summarize verified pass/fail "
            "limits as a table.",
            [],
        )

    limits = _load_limits_rows(rid)
    by_id: dict[str, Any] = {
        getattr(c, "chunk_id", ""): c
        for c in (chunks or [])
        if getattr(c, "chunk_id", None)
    }
    # Ensure limit source chunks are available for citation chips when possible.
    missing = [
        (getattr(r, "source_chunk_id", None) or "").strip()
        for r in limits
        if (getattr(r, "source_chunk_id", None) or "").strip()
        and (getattr(r, "source_chunk_id", None) or "").strip() not in by_id
    ]
    if missing:
        try:
            from retrieval.retrieve import get_qdrant_client
            from qdrant_client import models as qm
            import os

            client = get_qdrant_client()
            coll = os.getenv("QDRANT_COLLECTION", "regulations")
            for cid in missing:
                pts, _ = client.scroll(
                    collection_name=coll,
                    scroll_filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="chunk_id", match=qm.MatchValue(value=cid)
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if not pts:
                    continue
                from retrieval.retrieve import _payload_to_chunk

                chunk = _payload_to_chunk(pts[0].payload or {}, score=1.0)
                if chunk.chunk_id:
                    by_id[chunk.chunk_id] = chunk
        except Exception as exc:  # noqa: BLE001
            logger.debug("limits aggregation chunk fetch skipped: %s", exc)

    table_md, source_ids = format_limits_markdown_table(
        limits, regulation_id=rid, chunks_by_id=by_id
    )
    short = rid.replace("UN-ECE-", "UN ")
    intro = (
        f"Verified pass/fail / injury limits for **{short}** "
        f"(structured limits table — one row per criterion):\n\n"
    )
    sources: list[Any] = []
    if to_source is not None:
        for cid in source_ids:
            chunk = by_id.get(cid)
            if chunk is not None:
                sources.append(to_source(chunk))
    logger.info(
        "limits_aggregation rid=%s n_rows=%d n_sources=%d",
        rid,
        len(limits),
        len(sources),
    )
    return intro + table_md, sources
