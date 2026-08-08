"""CHECKLIST_GEN pipeline — per-category retrieval + structured cited checklist.

For \"generate a checklist for X testing/homologation\":

1. Resolve regulation(s) and fixed procedural categories (vehicle prep, dummy
   installation, instrumentation, injury criteria, documentation, …).
2. Run a SEPARATE retrieval per category (do not rely on one top-k to surface all).
3. Assemble a checklist grouped by category; each item cites a source clause.
4. Explicitly note categories with no retrieved content (silence ≠ not required).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "checklist_categories.json"
DEFAULT_PER_CATEGORY_TOP_K = 4

# True homologation/prep checklist cues — not bare "list every requirement…".
_CHECKLIST_PIPELINE_RE = re.compile(
    r"(?ix)\b("
    r"checklist|"
    r"homologat(?:e|ion|ing)|"
    r"prepar(?:e|ing|ation)\s+(?:a\s+)?(?:vehicle|test)|"
    r"steps\s+to\s+prepare|"
    r"generate\s+a\s+checklist|"
    r"preparation\s+(?:checklist|steps)"
    r")\b"
)

CHECKLIST_SYSTEM_PROMPT = """\
You are a UNECE passive-safety regulation assistant producing a TEST /
HOMOLOGATION PREPARATION CHECKLIST.

Reply with a single JSON object:
{
  "answer_segments": [
    {
      "text": "<one checklist item as an actionable requirement; no §/page/chips>",
      "citation_chunk_id": "<exact chunk_id from a provided passage>",
      "category_id": "<one of the category ids listed in the user message>"
    }
  ]
}

STRICT RULES:
1. Answer ONLY from the provided context passages. Do not invent administrative
   steps that are not supported by a passage.
2. Each segment is ONE checklist item with EXACTLY ONE citation_chunk_id and
   EXACTLY ONE category_id from the allowed list.
3. Prefer multiple items per category when the passages support them.
4. Do NOT write section numbers, page numbers, or citation chips in "text".
5. If a category has no supporting passages, omit items for that category —
   the backend will note it as missing. Never invent filler for empty categories.
6. If nothing is supported, return {"answer_segments": []}.
"""

CHECKLIST_USER_INSTRUCTION = """\
CHECKLIST GENERATION — structured deliverable:
- Emit answer_segments grouped by category_id (vehicle_prep, test_prep,
  dummy_installation, instrumentation, injury_criteria, documentation, …).
- Each item must be an actionable preparation/test requirement grounded in a
  retrieved passage via citation_chunk_id.
- Do not invent steps. Empty categories are reported by the backend.
"""


@dataclass
class ChecklistCategory:
    id: str
    label: str
    order: int
    retrieval_queries: list[str]
    topic_terms: list[str] = field(default_factory=list)


@dataclass
class ChecklistExpansion:
    question: str
    regulation_id: str | None
    categories: list[ChecklistCategory]
    config_path: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "regulation_id": self.regulation_id,
            "categories": [c.id for c in self.categories],
            "config_path": self.config_path,
        }


@dataclass
class CategoryRetrieval:
    category: ChecklistCategory
    chunks: list[Any] = field(default_factory=list)
    found: bool = False


@dataclass
class ChecklistRetrievalResult:
    chunks: list[Any]
    expansion: ChecklistExpansion
    by_category: list[CategoryRetrieval]
    covered_categories: list[str] = field(default_factory=list)
    missing_categories: list[str] = field(default_factory=list)
    chunk_category: dict[str, str] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "regulation_id": self.expansion.regulation_id,
            "covered_categories": list(self.covered_categories),
            "missing_categories": list(self.missing_categories),
            "per_category_counts": {
                row.category.id: len(row.chunks) for row in self.by_category
            },
            "chunk_category": dict(self.chunk_category),
            "n_chunks": len(self.chunks),
        }


def rebuild_checklist_result(
    question: str,
    chunks: Sequence[Any],
    *,
    meta: dict[str, Any] | None = None,
) -> ChecklistRetrievalResult:
    """Rebuild a ChecklistRetrievalResult from answer-time chunks + retrieve meta."""
    expansion = expand_checklist_query(question)
    meta = meta or {}
    chunk_category = {
        str(k): str(v)
        for k, v in (meta.get("chunk_category") or {}).items()
        if str(k).strip() and str(v).strip()
    }
    covered = [str(x) for x in (meta.get("covered_categories") or [])]
    missing = [str(x) for x in (meta.get("missing_categories") or [])]
    if not covered and not missing:
        # Infer from chunk_category labels when meta missing.
        present = set(chunk_category.values())
        covered = [c.id for c in expansion.categories if c.id in present]
        missing = [c.id for c in expansion.categories if c.id not in present]

    by_id = {getattr(c, "chunk_id", ""): c for c in chunks if getattr(c, "chunk_id", None)}
    by_category: list[CategoryRetrieval] = []
    for cat in expansion.categories:
        cat_chunks = [
            by_id[cid]
            for cid, label in chunk_category.items()
            if label == cat.id and cid in by_id
        ]
        # Also attach unlabelled chunks only to first category? Skip — keep empty.
        found = cat.id in covered and bool(cat_chunks)
        if cat.id in covered and not cat_chunks:
            # Covered at retrieve time but trimmed by budget — still "found" for notes?
            found = False
            if cat.id not in missing:
                missing = list(missing) + [cat.id]
            covered = [c for c in covered if c != cat.id]
        by_category.append(
            CategoryRetrieval(category=cat, chunks=cat_chunks, found=found)
        )

    return ChecklistRetrievalResult(
        chunks=list(chunks),
        expansion=expansion,
        by_category=by_category,
        covered_categories=covered,
        missing_categories=missing,
        chunk_category=chunk_category,
    )


_CONFIG_CACHE: tuple[float, list[ChecklistCategory], Path] | None = None


def is_checklist_pipeline_query(question: str) -> bool:
    """True for homologation/prep checklist asks (not bare enumerative lists)."""
    return bool(_CHECKLIST_PIPELINE_RE.search(question or ""))


def checklist_categories_path() -> Path:
    raw = (os.getenv("CHECKLIST_CATEGORIES_PATH") or "").strip()
    return Path(raw) if raw else DEFAULT_CONFIG_PATH


def load_checklist_categories(
    *, path: Path | None = None, force: bool = False
) -> list[ChecklistCategory]:
    global _CONFIG_CACHE
    cfg = path or checklist_categories_path()
    try:
        mtime = cfg.stat().st_mtime
    except OSError:
        logger.warning("checklist_categories config missing: %s", cfg)
        return []
    if (
        not force
        and _CONFIG_CACHE is not None
        and _CONFIG_CACHE[0] == mtime
        and _CONFIG_CACHE[2] == cfg
    ):
        return list(_CONFIG_CACHE[1])

    data = json.loads(cfg.read_text(encoding="utf-8"))
    cats: list[ChecklistCategory] = []
    for row in data.get("categories") or []:
        cats.append(
            ChecklistCategory(
                id=str(row.get("id") or "").strip(),
                label=str(row.get("label") or row.get("id") or "").strip(),
                order=int(row.get("order") or 99),
                retrieval_queries=[
                    str(q).strip()
                    for q in (row.get("retrieval_queries") or [])
                    if str(q).strip()
                ],
                topic_terms=[
                    str(t).strip().lower()
                    for t in (row.get("topic_terms") or [])
                    if str(t).strip()
                ],
            )
        )
    cats.sort(key=lambda c: (c.order, c.id))
    _CONFIG_CACHE = (mtime, cats, cfg)
    return list(cats)


def expand_checklist_query(question: str) -> ChecklistExpansion:
    """Identify regulation + checklist categories before retrieval."""
    from retrieval.enumerative import detect_named_regulation, resolve_hard_regulation_filter

    q = (question or "").strip()
    named = resolve_hard_regulation_filter(q) or detect_named_regulation(q)
    cats = load_checklist_categories()
    return ChecklistExpansion(
        question=q,
        regulation_id=named,
        categories=cats,
        config_path=str(checklist_categories_path()),
    )


def per_category_top_k() -> int:
    try:
        return max(
            1,
            int(
                (os.getenv("CHECKLIST_PER_CATEGORY_TOP_K") or str(DEFAULT_PER_CATEGORY_TOP_K)).strip()
            ),
        )
    except ValueError:
        return DEFAULT_PER_CATEGORY_TOP_K


def _chunk_matches_category(chunk: Any, category: ChecklistCategory) -> bool:
    blob = " ".join(
        [
            getattr(chunk, "text", "") or "",
            getattr(chunk, "section_title", "") or "",
            getattr(chunk, "section_number", "") or "",
        ]
    ).lower()
    if not blob:
        return False
    terms = category.topic_terms
    if not terms:
        return True
    hits = sum(1 for t in terms if t and t in blob)
    need = 1 if len(terms) <= 3 else max(1, min(2, len(terms) // 3))
    return hits >= need


def retrieve_checklist(
    query: str,
    *,
    top_k_per_category: int | None = None,
    client: object | None = None,
    embedder: object | None = None,
    collection: str | None = None,
    do_rerank: bool = True,
    expansion: ChecklistExpansion | None = None,
) -> ChecklistRetrievalResult:
    """Separate hybrid retrieval per checklist category, then merge."""
    from retrieval.retrieve import hybrid_search

    expansion = expansion or expand_checklist_query(query)
    per_k = top_k_per_category or per_category_top_k()
    reg = expansion.regulation_id

    by_category: list[CategoryRetrieval] = []
    merged: list[Any] = []
    chunk_category: dict[str, str] = {}
    seen_ids: set[str] = set()
    covered: list[str] = []
    missing: list[str] = []

    for cat in expansion.categories:
        hits: list[Any] = []
        queries = list(cat.retrieval_queries) or [f"{cat.label} {query}"]
        # Always include a category-label probe tied to the user question.
        queries = queries + [f"{cat.label} {expansion.question}"]
        for sq in queries[:4]:
            try:
                batch = hybrid_search(
                    sq,
                    top_k=max(per_k * 2, 8),
                    regulation_id=reg,
                    client=client,  # type: ignore[arg-type]
                    embedder=embedder,  # type: ignore[arg-type]
                    collection=collection,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "checklist hybrid_search failed category=%s: %s", cat.id, exc
                )
                batch = []
            for c in batch:
                cid = getattr(c, "chunk_id", None) or ""
                # Allow same chunk in multiple categories only once in merged set,
                # but first category wins for labelling.
                if cid and cid in seen_ids:
                    continue
                hits.append(c)

        if do_rerank and hits:
            try:
                from retrieval.rerank import rerank as _rerank

                probe = f"{cat.label} {expansion.question}"
                hits = _rerank(probe, hits, top_n=max(per_k * 2, len(hits)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("checklist rerank skipped category=%s: %s", cat.id, exc)

        relevant = [c for c in hits if _chunk_matches_category(c, cat)]
        chosen = (relevant or hits)[:per_k]
        found = bool(relevant) or (bool(chosen) and bool(hits))
        # Prefer topic-matched; if only weak hits, still keep them but mark found
        # only when topic matched OR we have substantive text.
        if relevant:
            covered.append(cat.id)
            found = True
        else:
            missing.append(cat.id)
            found = False
            # Keep weak hits for context only when nothing better exists — still
            # report category as missing so the engineer sees incompleteness.
            chosen = []

        row = CategoryRetrieval(category=cat, chunks=list(chosen), found=found)
        by_category.append(row)
        for c in chosen:
            cid = getattr(c, "chunk_id", None) or ""
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                chunk_category[cid] = cat.id
                merged.append(c)

        logger.info(
            "checklist category=%s found=%s chunks=%d queries=%d reg=%s",
            cat.id,
            found,
            len(chosen),
            len(queries),
            reg,
        )

    result = ChecklistRetrievalResult(
        chunks=merged,
        expansion=expansion,
        by_category=by_category,
        covered_categories=covered,
        missing_categories=missing,
        chunk_category=chunk_category,
    )
    logger.info(
        "checklist retrieve reg=%s covered=%s missing=%s total_chunks=%d",
        reg,
        covered,
        missing,
        len(merged),
    )
    try:
        from observability.context import get_current_trace

        tr = get_current_trace()
        if tr is not None:
            tr.optimizations["checklist_gen"] = True
            tr.optimizations["checklist_retrieval"] = result.to_public_dict()
    except Exception:  # noqa: BLE001
        pass
    return result


def format_checklist_context(
    result: ChecklistRetrievalResult,
    *,
    chunks: Sequence[Any] | None = None,
) -> str:
    """Group passages by category for the LLM; list empty categories explicitly."""
    pool = list(chunks) if chunks is not None else list(result.chunks)
    by_id = {getattr(c, "chunk_id", ""): c for c in pool if getattr(c, "chunk_id", None)}
    parts: list[str] = []
    idx = 0
    for row in result.by_category:
        parts.append(f"## Category `{row.category.id}` — {row.category.label}")
        if not row.found or not row.chunks:
            parts.append(
                "(No retrieved passages for this category — do not invent items.)"
            )
            continue
        for c in row.chunks:
            cid = getattr(c, "chunk_id", "") or ""
            # Prefer budget-trimmed copy from ``chunks`` when present.
            use = by_id.get(cid, c)
            idx += 1
            if hasattr(use, "context_block"):
                block = use.context_block(index=idx)
            else:
                block = f"[passage {idx}] chunk_id={cid}\n{getattr(use, 'text', '')}"
            parts.append(f"category_id={row.category.id}\n{block}")
    if result.missing_categories:
        labels = []
        for row in result.by_category:
            if row.category.id in result.missing_categories:
                labels.append(f"{row.category.label} (`{row.category.id}`)")
        parts.append(
            "## Categories with no retrieved content\n"
            + ", ".join(labels)
            + "\n(Backend will note these as incomplete — do not fabricate items.)"
        )
    return "\n\n".join(parts)


def checklist_answer_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "checklist_gen_answer",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "answer_segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "citation_chunk_id": {"type": "string"},
                                "category_id": {"type": "string"},
                            },
                            "required": ["text", "citation_chunk_id", "category_id"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["answer_segments"],
                "additionalProperties": False,
            },
        },
    }


def keep_grounded_checklist_segments(
    segments: Sequence[Any],
    allowed_ids: set[str],
    *,
    allowed_categories: set[str] | None = None,
) -> tuple[list[Any], list[str]]:
    """Per-item grounding: keep segments with valid chunk ids (and category)."""
    kept: list[Any] = []
    dropped: list[str] = []
    for seg in segments:
        cid = (getattr(seg, "citation_chunk_id", None) or "").strip()
        cat = (getattr(seg, "category_id", None) or getattr(seg, "claim_kind", None) or "").strip()
        if not cid or cid not in allowed_ids:
            dropped.append(cid or "(empty)")
            continue
        if allowed_categories and cat and cat not in allowed_categories:
            # Remap unknown category via chunk map later; still keep if cited.
            pass
        # Stash category_id on segment when model put it in claim_kind by mistake.
        if hasattr(seg, "category_id") and not (getattr(seg, "category_id", None) or "").strip():
            try:
                seg.category_id = cat  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        kept.append(seg)
    return kept, dropped


def render_checklist_answer(
    segments: Sequence[Any],
    chunks_by_id: dict[str, Any],
    *,
    expansion: ChecklistExpansion,
    covered: Sequence[str],
    missing: Sequence[str],
    chunk_category: dict[str, str] | None = None,
    to_source: Any,
) -> tuple[str, list[Any]]:
    """Grouped checklist with explicit empty-category notes."""
    from generation.answer import _CITATION_CHIP_RE

    chunk_category = chunk_category or {}
    cat_by_id = {c.id: c for c in expansion.categories}
    # Bucket segments by category.
    buckets: dict[str, list[str]] = {c.id: [] for c in expansion.categories}
    sources: list[Any] = []
    seen: set[str] = set()

    for seg in segments:
        cid = (getattr(seg, "citation_chunk_id", None) or "").strip()
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        chip = chunk.citation_tag()
        body = (getattr(seg, "text", None) or "").strip()
        body = _CITATION_CHIP_RE.sub("", body).strip()
        cat_id = (
            (getattr(seg, "category_id", None) or "").strip()
            or chunk_category.get(cid, "")
            or "vehicle_prep"
        )
        if cat_id not in buckets:
            buckets[cat_id] = []
        line = f"- [ ] {body} {chip}".strip()
        buckets[cat_id].append(line)
        if cid not in seen:
            seen.add(cid)
            sources.append(to_source(chunk))

    reg = expansion.regulation_id or "indexed regulation(s)"
    title = f"Checklist for {reg} testing / homologation preparation"
    blocks: list[str] = [title, ""]

    for cat in expansion.categories:
        blocks.append(f"## {cat.label}")
        items = buckets.get(cat.id) or []
        if items:
            blocks.extend(items)
        else:
            blocks.append(
                "_No content found in the indexed regulation for this category — "
                "the checklist may be incomplete; do not assume this means "
                "'not required'._"
            )
        blocks.append("")

    # Any leftover category ids not in config.
    for cat_id, items in buckets.items():
        if cat_id in cat_by_id or not items:
            continue
        blocks.append(f"## {cat_id}")
        blocks.extend(items)
        blocks.append("")

    if missing:
        labels = [
            cat_by_id[m].label if m in cat_by_id else m for m in missing
        ]
        blocks.append("### Incomplete categories")
        blocks.append(
            "No indexed content was retrieved for: "
            + ", ".join(labels)
            + ". Review the source regulation (or ingest missing annexes) before "
            "treating this checklist as complete."
        )

    return "\n".join(blocks).strip(), sources


def extractive_checklist_fallback(
    *,
    result: ChecklistRetrievalResult,
    chunks: Sequence[Any],
    to_source: Any,
    max_items_per_category: int = 3,
) -> tuple[str, list[Any]]:
    """Deterministic checklist from per-category chunks when the LLM abstains."""
    from generation.answer import AnswerSegment

    segs: list[AnswerSegment] = []
    by_id = {getattr(c, "chunk_id", ""): c for c in chunks if getattr(c, "chunk_id", None)}
    for row in result.by_category:
        if not row.found:
            continue
        for c in row.chunks[:max_items_per_category]:
            cid = getattr(c, "chunk_id", "") or ""
            use = by_id.get(cid, c)
            text = " ".join((getattr(use, "text", None) or "").split())
            if not text:
                continue
            words = text.split()
            excerpt = " ".join(words[:35]) + ("…" if len(words) > 35 else "")
            segs.append(
                AnswerSegment(
                    text=excerpt,
                    citation_chunk_id=cid,
                    category_id=row.category.id,
                )
            )
    if not segs and not result.missing_categories:
        return "", []
    # Even with zero segments, render empty-category notes when categories missing.
    return render_checklist_answer(
        segs,
        by_id,
        expansion=result.expansion,
        covered=result.covered_categories,
        missing=result.missing_categories,
        chunk_category=result.chunk_category,
        to_source=to_source,
    )
