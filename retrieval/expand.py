"""Small-to-big: expand reranked leaf chunks to parent sections."""

from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ingestion.embed_upsert import DEFAULT_COLLECTION
from retrieval.context_budget import approx_tokens, max_parent_expand_tokens
from retrieval.retrieve import RetrievedChunk, _payload_to_chunk

logger = logging.getLogger(__name__)

# Sibling scroll cap — Annex parents can have dozens of children; we must not
# pull an unbounded set into one merged blob.
_DEFAULT_SCROLL_LIMIT = 16


def _scroll_limit() -> int:
    try:
        return max(
            4,
            int((os.getenv("SMALL_TO_BIG_SCROLL_LIMIT") or str(_DEFAULT_SCROLL_LIMIT)).strip()),
        )
    except ValueError:
        return _DEFAULT_SCROLL_LIMIT


def _scroll_filter(
    client: QdrantClient,
    *,
    collection: str,
    key: str,
    value: str,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    if not value:
        return []
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qm.Filter(
            must=[qm.FieldCondition(key=key, match=qm.MatchValue(value=value))]
        ),
        limit=limit if limit is not None else _scroll_limit(),
        with_payload=True,
        with_vectors=False,
    )
    return [_payload_to_chunk(p.payload or {}, 0.0) for p in points]


def _merge_section(
    chunks: list[RetrievedChunk],
    *,
    score: float,
    section_id: str,
    cite_from: RetrievedChunk | None = None,
) -> RetrievedChunk:
    """Merge sibling/parent payloads into one expanded section chunk.

    Citation metadata (section_number / page / bbox) prefers the triggering leaf
    so small-to-big does not rewrite a precise hit like 5.2.7 into bare 5.2.
    """
    chunks = [c for c in chunks if (c.text or "").strip()]
    if not chunks:
        raise ValueError("no chunks to merge")
    # Prefer the exact section_id match as the shell; else first by section_number depth.
    head = next((c for c in chunks if c.section_id == section_id), chunks[0])
    cite = cite_from or head
    # Stable order: shorter section numbers first, then page.
    ordered = sorted(
        chunks,
        key=lambda c: (c.section_number or "", c.page_number or 0, c.chunk_id),
    )
    texts: list[str] = []
    seen_text: set[str] = set()
    bboxes: list[list[float]] = []
    pages: list[int] = []
    for c in ordered:
        t = (c.text or "").strip()
        if t and t not in seen_text:
            seen_text.add(t)
            texts.append(t)
        if c.bounding_box:
            bboxes.append(c.bounding_box)
        if c.page_number is not None:
            pages.append(c.page_number)

    return head.model_copy(
        update={
            "chunk_id": f"expanded::{section_id}",
            "section_id": section_id,
            "section_number": cite.section_number or head.section_number,
            "section_title": cite.section_title or head.section_title,
            "text": "\n\n".join(texts),
            "enriched_text": "",
            "page_number": (
                cite.page_number
                if cite.page_number is not None
                else (pages[0] if pages else head.page_number)
            ),
            "bounding_box": cite.bounding_box or (bboxes[0] if bboxes else head.bounding_box),
            "score": score,
            "content_type": "clause",
        }
    )


def expand_to_parents(
    chunks: list[RetrievedChunk],
    *,
    client: QdrantClient,
    collection: str = DEFAULT_COLLECTION,
) -> list[RetrievedChunk]:
    """Replace each leaf with its parent section; dedupe parents.

    Lookup order for a leaf:
      1. points whose ``section_id`` == leaf.parent_section_id
      2. else all siblings sharing that parent_section_id
      3. else the leaf itself

    If the merged parent would exceed ``MAX_PARENT_EXPAND_TOKENS``, keep the
    smaller leaf with its own citation (Annex-scale parents must not flood the LLM).
    """
    if not chunks:
        return []

    max_parent_tok = max_parent_expand_tokens()
    expanded: list[RetrievedChunk] = []
    seen_parents: set[str] = set()
    skipped_oversized = 0

    for leaf in chunks:
        parent_key = (leaf.parent_section_id or leaf.section_id or leaf.chunk_id).strip()
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)

        parents = _scroll_filter(
            client, collection=collection, key="section_id", value=parent_key
        )
        # Always pull children that name this section as their parent. Looking up
        # section_id alone often returns only a short shell (e.g. 5.2.8.1.4 intro)
        # and drops numeric leaves (5.2.8.1.4.1 / .4.2) — which then forces the
        # answer path to abstain as grounding_rejected despite high retrieval scores.
        children = _scroll_filter(
            client,
            collection=collection,
            key="parent_section_id",
            value=parent_key,
        )
        by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in parents}
        for c in children:
            by_id[c.chunk_id] = c
        by_id[leaf.chunk_id] = leaf
        parents = list(by_id.values())

        # Legacy fallback when the parent_key itself was a leaf with no children.
        if len(parents) <= 1 and leaf.parent_section_id and leaf.parent_section_id != parent_key:
            siblings = _scroll_filter(
                client,
                collection=collection,
                key="parent_section_id",
                value=leaf.parent_section_id,
            )
            for c in siblings:
                by_id[c.chunk_id] = c
            by_id[leaf.chunk_id] = leaf
            parents = list(by_id.values())

        if parents:
            try:
                merged = _merge_section(
                    parents,
                    score=leaf.score,
                    section_id=parent_key,
                    cite_from=leaf,
                )
                parent_tok = approx_tokens(merged.text)
                if parent_tok > max_parent_tok:
                    skipped_oversized += 1
                    logger.info(
                        "small-to-big: skip oversized parent %s (~%d tok > %d); keep leaf %s",
                        parent_key,
                        parent_tok,
                        max_parent_tok,
                        leaf.chunk_id,
                    )
                    expanded.append(leaf)
                else:
                    expanded.append(merged)
                continue
            except ValueError:
                pass

        expanded.append(leaf)

    logger.info(
        "small-to-big: %d leaves → %d sections (skipped_oversized_parents=%d, max_parent_tok=%d)",
        len(chunks),
        len(expanded),
        skipped_oversized,
        max_parent_tok,
    )
    return expanded
