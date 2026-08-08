"""Hybrid dense + BM25 retrieval with RRF, rewrite, rerank, small-to-big."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ingestion.embed_upsert import (
    BM25_MODEL,
    DEFAULT_COLLECTION,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    Embedder,
    get_qdrant_client,
)

logger = logging.getLogger(__name__)

DEFAULT_HYBRID_TOP_K = 30
DEFAULT_PREFETCH = 30

_ROOT = Path(__file__).resolve().parents[1]
_SCOPE_CONFIG_PATH = _ROOT / "config" / "scope_sections.json"
_DEFINITIONS_CONFIG_PATH = _ROOT / "config" / "definitions_sections.json"

# Objective / scope / purpose queries — regex/keyword, no LLM.
_SCOPE_QUERY_RE = re.compile(
    r"(?ix)"
    r"("
    r"\bobjectives?\b"
    r"|\bscopes?\b"
    r"|\bpurposes?\b"
    r"|\bwhat\s+is\s+this\s+regulation\s+about\b"
    r"|\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:objective|scope|purpose)\b"
    r"|\bwhat\s+does\s+(?:this\s+)?regulation\s+(?:cover|address|apply(?:\s+to)?|concern)\b"
    r"|\bwhat\s+is\s+(?:UN[- ]?)?R?\d+\s+about\b"
    r"|\babout\s+this\s+regulation\b"
    r"|\b(?:which|what)\s+vehicles?\s+(?:are\s+)?covered\b"
    r"|\bvehicles?\s+covered\s+under\b"
    r")"
)

# "Define X" / definition-of queries → prepend Definitions article.
_DEFINE_QUERY_RE = re.compile(
    r"(?ix)"
    r"("
    r"^\s*define\b"
    r"|\bdefinition\s+of\b"
    r"|\bwhat\s+is\s+the\s+definition\s+of\b"
    r"|\bhow\s+(?:is|are)\b.+\bdefined\b"
    r")"
)

_scope_map_cache: dict[str, str] | None = None
_scope_map_lock = threading.Lock()
_definitions_map_cache: dict[str, str] | None = None
_definitions_map_lock = threading.Lock()

# --- Live indexed-regulation catalog (invalidated on ingest) ----------------

_catalog_lock = threading.Lock()
_catalog_cache: list["IndexedRegulation"] | None = None
_catalog_cache_version: int = -1


@dataclass(frozen=True)
class IndexedRegulation:
    regulation_id: str
    revision: str
    chunk_count: int

    @property
    def label(self) -> str:
        """Short UI / message label, e.g. ``R94 (Rev.3)``."""
        short = (self.regulation_id or "").replace("UN-ECE-", "") or self.regulation_id
        rev = (self.revision or "").strip()
        if rev and rev not in {"—", "-", "?"}:
            return f"{short} ({rev})"
        return short


def invalidate_indexed_regulations_cache() -> None:
    """Clear the in-memory catalog (call after upsert / delete / upload)."""
    global _catalog_cache, _catalog_cache_version
    with _catalog_lock:
        _catalog_cache = None
        _catalog_cache_version = -1
    logger.info("Indexed-regulations cache invalidated")


def get_indexed_regulations(
    *,
    client: QdrantClient | None = None,
    collection: str | None = None,
    force_refresh: bool = False,
) -> list[IndexedRegulation]:
    """Distinct ``regulation_id`` + ``revision`` currently in Qdrant (cached)."""
    global _catalog_cache, _catalog_cache_version

    try:
        from api.cache_version import get_cache_version

        current_ver = get_cache_version()
    except Exception:  # noqa: BLE001
        current_ver = 0

    with _catalog_lock:
        if (
            _catalog_cache is not None
            and not force_refresh
            and _catalog_cache_version == current_ver
        ):
            return list(_catalog_cache)

    load_dotenv()
    collection = collection or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    client = client or get_qdrant_client()

    try:
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list Qdrant collections: %s", exc)
        return []

    if collection not in existing:
        with _catalog_lock:
            _catalog_cache = []
            _catalog_cache_version = current_ver
        return []

    counts: dict[tuple[str, str], int] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=["regulation_id", "revision"],
            with_vectors=False,
        )
        for pt in points:
            pl = pt.payload or {}
            rid = str(pl.get("regulation_id") or "").strip() or "(unknown)"
            rev = str(pl.get("revision") or "").strip()
            key = (rid, rev)
            counts[key] = counts.get(key, 0) + 1
        if offset is None:
            break

    rows = [
        IndexedRegulation(regulation_id=rid, revision=rev, chunk_count=n)
        for (rid, rev), n in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]
    with _catalog_lock:
        _catalog_cache = rows
        _catalog_cache_version = current_ver
    return list(rows)


def indexed_regulation_ids() -> set[str]:
    return {r.regulation_id for r in get_indexed_regulations()}


def format_indexed_regulations_label(regs: Sequence[IndexedRegulation] | None = None) -> str:
    rows = list(regs) if regs is not None else get_indexed_regulations()
    if not rows:
        return "(none indexed yet)"
    # Dedupe by regulation_id for the short message (prefer first revision label).
    seen: set[str] = set()
    labels: list[str] = []
    for r in rows:
        if r.regulation_id in seen:
            continue
        seen.add(r.regulation_id)
        labels.append(r.label)
    return " / ".join(labels)


def load_scope_sections(path: Path | None = None) -> dict[str, str]:
    """Per-regulation Scope/Purpose ``section_number`` map (cached)."""
    return _load_section_map(
        path or _SCOPE_CONFIG_PATH,
        cache_attr="_scope_map_cache",
        lock=_scope_map_lock,
        defaults={"UN-ECE-R94": "1", "UN-ECE-R95": "1"},
        label="Scope",
    )


def load_definitions_sections(path: Path | None = None) -> dict[str, str]:
    """Per-regulation Definitions article ``section_number`` map (cached)."""
    return _load_section_map(
        path or _DEFINITIONS_CONFIG_PATH,
        cache_attr="_definitions_map_cache",
        lock=_definitions_map_lock,
        defaults={"UN-ECE-R94": "2", "UN-ECE-R95": "2"},
        label="Definitions",
    )


def _load_section_map(
    cfg_path: Path,
    *,
    cache_attr: str,
    lock: threading.Lock,
    defaults: dict[str, str],
    label: str,
) -> dict[str, str]:
    global _scope_map_cache, _definitions_map_cache
    with lock:
        cached = _scope_map_cache if cache_attr == "_scope_map_cache" else _definitions_map_cache
        # Only use module cache when loading the default path.
        if cached is not None and cfg_path in {_SCOPE_CONFIG_PATH, _DEFINITIONS_CONFIG_PATH}:
            # Re-check which cache — simpler to just always read file if path overridden.
            pass
        if cache_attr == "_scope_map_cache" and _scope_map_cache is not None and cfg_path == _SCOPE_CONFIG_PATH:
            return dict(_scope_map_cache)
        if (
            cache_attr == "_definitions_map_cache"
            and _definitions_map_cache is not None
            and cfg_path == _DEFINITIONS_CONFIG_PATH
        ):
            return dict(_definitions_map_cache)

    data: dict[str, str] = {}
    if cfg_path.is_file():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k, v in raw.items():
                if str(k).startswith("_"):
                    continue
                sec = str(v).strip()
                if sec:
                    data[str(k).strip()] = sec
    else:
        logger.warning("%s config missing at %s — using defaults", label, cfg_path)
        data = dict(defaults)

    with lock:
        if cache_attr == "_scope_map_cache" and cfg_path == _SCOPE_CONFIG_PATH:
            _scope_map_cache = data
        elif cache_attr == "_definitions_map_cache" and cfg_path == _DEFINITIONS_CONFIG_PATH:
            _definitions_map_cache = data
    return dict(data)


def is_scope_objective_query(query: str) -> bool:
    """True for objective/scope/purpose / 'what is this regulation about' queries."""
    q = (query or "").strip()
    if not q:
        return False
    return _SCOPE_QUERY_RE.search(q) is not None


def is_definition_query(query: str) -> bool:
    """True for 'Define X' / 'definition of' style queries."""
    q = (query or "").strip()
    if not q:
        return False
    return _DEFINE_QUERY_RE.search(q) is not None


def fetch_section_chunks(
    *,
    section_map: dict[str, str],
    regulation_id: str | None = None,
    client: QdrantClient | None = None,
    collection: str | None = None,
    label: str = "section",
) -> list[RetrievedChunk]:
    """Scroll Qdrant for configured article chunks (Scope, Definitions, …)."""
    load_dotenv()
    collection = collection or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    client = client or get_qdrant_client()

    targets: list[tuple[str, str]] = []
    if regulation_id:
        sec = section_map.get(regulation_id)
        if sec:
            targets.append((regulation_id, sec))
    else:
        indexed = indexed_regulation_ids()
        for rid, sec in section_map.items():
            if not indexed or rid in indexed:
                targets.append((rid, sec))

    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    for rid, sec in targets:
        must = [
            qm.FieldCondition(key="regulation_id", match=qm.MatchValue(value=rid)),
            qm.FieldCondition(key="section_number", match=qm.MatchValue(value=sec)),
        ]
        try:
            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter=qm.Filter(must=must),
                limit=8,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s scroll failed for %s §%s: %s", label, rid, sec, exc)
            continue
        for pt in points:
            chunk = _payload_to_chunk(pt.payload or {}, score=1.0)
            if not chunk.chunk_id or chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            out.append(chunk)
    logger.info(
        "%s fetch regs=%s chunks=%d ids=%s",
        label,
        [t[0] for t in targets],
        len(out),
        [c.chunk_id for c in out],
    )
    return out


def fetch_scope_chunks(
    *,
    regulation_id: str | None = None,
    client: QdrantClient | None = None,
    collection: str | None = None,
    scope_map: dict[str, str] | None = None,
) -> list[RetrievedChunk]:
    """Scroll Qdrant for Scope/Purpose article chunks from ``config/scope_sections.json``."""
    mapping = scope_map if scope_map is not None else load_scope_sections()
    return fetch_section_chunks(
        section_map=mapping,
        regulation_id=regulation_id,
        client=client,
        collection=collection,
        label="scope",
    )


def fetch_definitions_chunks(
    *,
    regulation_id: str | None = None,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> list[RetrievedChunk]:
    """Scroll Qdrant for Definitions article chunks from ``config/definitions_sections.json``."""
    return fetch_section_chunks(
        section_map=load_definitions_sections(),
        regulation_id=regulation_id,
        client=client,
        collection=collection,
        label="definitions",
    )


def fetch_chunks_by_ids(
    chunk_ids: Sequence[str],
    *,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> list[RetrievedChunk]:
    """Load specific chunks by ``chunk_id`` (order preserved)."""
    load_dotenv()
    collection = collection or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    client = client or get_qdrant_client()
    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    for cid in chunk_ids:
        cid = str(cid).strip()
        if not cid or cid in seen:
            continue
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=cid))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            continue
        chunk = _payload_to_chunk(points[0].payload or {}, score=1.0)
        seen.add(cid)
        out.append(chunk)
    return out


def prepend_unique_chunks(
    leading: Sequence[RetrievedChunk],
    candidates: Sequence[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Put ``leading`` first; drop duplicates from the hybrid tail."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for chunk in list(leading) + list(candidates):
        key = chunk.chunk_id or chunk.section_id or str(id(chunk))
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


# Back-compat alias
prepend_scope_chunks = prepend_unique_chunks


class RetrievedChunk(BaseModel):
    """One hit from hybrid search, with full grounding metadata."""

    chunk_id: str
    text: str
    enriched_text: str = ""
    regulation_id: str = ""
    revision: str = ""
    section_number: str = ""
    section_title: str = ""
    page_number: int | None = None
    bounding_box: list[float] = Field(default_factory=list)
    content_type: str = "clause"
    parent_section_id: str | None = None
    section_id: str = ""
    heading_path: list[str] = Field(default_factory=list)
    score: float = 0.0

    def citation_tag(self) -> str:
        """Inline citation form: [regulation_id §section_number, p.page]."""
        sec = self.section_number or "?"
        page = self.page_number if self.page_number is not None else "?"
        reg = self.regulation_id or "?"
        return f"[{reg} §{sec}, p.{page}]"

    def context_block(self, *, index: int) -> str:
        """Passage block fed to the LLM (cite by chunk_id only)."""
        page = self.page_number if self.page_number is not None else "?"
        header = (
            f"[passage {index}] chunk_id={self.chunk_id} "
            f"regulation_id={self.regulation_id or '?'} "
            f"(backend metadata: section_number={self.section_number or '?'} "
            f"page={page} — do not write section/page numbers in your text; "
            f"cite only via citation_chunk_id={self.chunk_id}) "
            f"{self.section_title}".strip()
        )
        body = (self.text or self.enriched_text or "").strip()
        return f"{header}\n{body}"


def _payload_to_chunk(payload: dict[str, Any], score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(payload.get("chunk_id") or ""),
        text=str(payload.get("text") or ""),
        enriched_text=str(payload.get("enriched_text") or ""),
        regulation_id=str(payload.get("regulation_id") or ""),
        revision=str(payload.get("revision") or ""),
        section_number=str(payload.get("section_number") or ""),
        section_title=str(payload.get("section_title") or ""),
        page_number=payload.get("page_number"),
        bounding_box=list(payload.get("bounding_box") or []),
        content_type=str(payload.get("content_type") or "clause"),
        parent_section_id=payload.get("parent_section_id"),
        section_id=str(payload.get("section_id") or ""),
        heading_path=list(payload.get("heading_path") or []),
        score=float(score),
    )


def _regulation_filter(regulation_id: str | None) -> qm.Filter | None:
    if not regulation_id:
        return None
    return qm.Filter(
        must=[
            qm.FieldCondition(
                key="regulation_id",
                match=qm.MatchValue(value=regulation_id),
            )
        ]
    )


def rrf_merge(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    top_k: int = DEFAULT_HYBRID_TOP_K,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Client-side reciprocal rank fusion across multiple result lists.

    Optional ``weights`` scales each list's contribution (e.g. raise BM25
    weight when the query contains an exact regulatory term like HPC).
    """
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for list_i, ranked in enumerate(ranked_lists):
        w = 1.0
        if weights is not None and list_i < len(weights):
            try:
                w = float(weights[list_i])
            except (TypeError, ValueError):
                w = 1.0
            w = max(0.0, w)
        for rank, chunk in enumerate(ranked, start=1):
            key = chunk.chunk_id or chunk.section_id or str(id(chunk))
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            prev = best.get(key)
            if prev is None or chunk.score > prev.score:
                best[key] = chunk
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[RetrievedChunk] = []
    for key, fused in ordered[:top_k]:
        out.append(best[key].model_copy(update={"score": fused}))
    return out


def _hybrid_fusion_weights(query: str) -> tuple[float, float]:
    """``(dense_weight, sparse_weight)`` for client-side RRF.

    Exact regulatory-term queries (HPC, electrolyte, …) raise the BM25/sparse
    weight so literal keyword hits beat dense near-misses (e.g. isolation-
    resistance neighbors when the ask is electrolyte spillage).
    """
    from retrieval.value_limit import is_exact_term_boost_query

    dense = float(os.getenv("HYBRID_DENSE_WEIGHT", "1.0") or "1.0")
    sparse = float(os.getenv("HYBRID_SPARSE_WEIGHT", "1.0") or "1.0")
    if is_exact_term_boost_query(query):
        dense = float(os.getenv("HYBRID_DENSE_WEIGHT_EXACT", "1.0") or "1.0")
        sparse = float(os.getenv("HYBRID_SPARSE_WEIGHT_EXACT", "1.75") or "1.75")
    return dense, sparse


def _qdrant_search_params() -> qm.SearchParams:
    """HNSW search params — raise ``ef`` so identical queries stay stable.

    Very low default ``ef`` can return different top-k under concurrent load.
    ``QDRANT_EXACT_SEARCH=1`` forces a full scan (deterministic; fine for our
    small regulation corpus).
    """
    exact = (os.getenv("QDRANT_EXACT_SEARCH") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        ef = int((os.getenv("QDRANT_HNSW_EF") or "256").strip() or "256")
    except ValueError:
        ef = 256
    return qm.SearchParams(hnsw_ef=max(16, ef), exact=exact)


def hybrid_search(
    query: str,
    *,
    top_k: int | None = None,
    prefetch_k: int | None = None,
    regulation_id: str | None = None,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
    collection: str | None = None,
) -> list[RetrievedChunk]:
    """Dense + BM25 sparse search fused with weighted RRF. Returns ~top_k (30).

    When the query names an injury criterion (HPC, RDC, …), BM25/sparse is
    weighted above dense so literal ``Head Performance Criterion`` matches beat
    dense near-misses (electrical-safety / energy-absorption neighbors).
    """
    load_dotenv()
    query = (query or "").strip()
    if not query:
        return []

    # Deterministic acronym expand for embedding + BM25 (never an LLM call).
    from retrieval.acronyms import expand_acronyms

    original_q = query
    query = expand_acronyms(query)
    acronym_fired = query != original_q
    logger.info(
        "hybrid expanded_query=%r original=%r acronym_expand=%s",
        query,
        original_q,
        acronym_fired,
    )

    top_k = top_k or int(os.getenv("HYBRID_TOP_K", str(DEFAULT_HYBRID_TOP_K)))
    prefetch_k = prefetch_k or int(os.getenv("HYBRID_PREFETCH_K", str(DEFAULT_PREFETCH)))
    collection = collection or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    client = client or get_qdrant_client()
    embedder = embedder or Embedder()

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        logger.warning("Qdrant collection %r missing — ingest a PDF first", collection)
        return []

    dense = embedder.embed([query])[0]
    qfilter = _regulation_filter(regulation_id)
    search_params = _qdrant_search_params()
    dense_w, sparse_w = _hybrid_fusion_weights(query)

    try:
        # Separate prefetches so we can weight BM25 above dense for exact terms.
        # (Qdrant's built-in Fusion.RRF is equal-weight only.)
        dense_resp = client.query_points(
            collection_name=collection,
            query=dense,
            using=DENSE_VECTOR,
            query_filter=qfilter,
            limit=prefetch_k,
            with_payload=True,
            search_params=search_params,
        )
        sparse_resp = client.query_points(
            collection_name=collection,
            query=qm.Document(text=query, model=BM25_MODEL),
            using=SPARSE_VECTOR,
            query_filter=qfilter,
            limit=prefetch_k,
            with_payload=True,
        )
        dense_chunks = [
            _payload_to_chunk(hit.payload or {}, float(hit.score or 0.0))
            for hit in dense_resp.points
        ]
        sparse_chunks = [
            _payload_to_chunk(hit.payload or {}, float(hit.score or 0.0))
            for hit in sparse_resp.points
        ]
        results = rrf_merge(
            [dense_chunks, sparse_chunks],
            top_k=top_k,
            weights=[dense_w, sparse_w],
        )
        logger.info(
            "hybrid weighted-RRF dense_w=%.2f sparse_w=%.2f q=%r hits=%d",
            dense_w,
            sparse_w,
            query[:80],
            len(results),
        )
    except Exception as exc:  # noqa: BLE001 — fall back to dense-only
        logger.warning("Hybrid RRF failed (%s); falling back to dense-only", exc)
        response = client.query_points(
            collection_name=collection,
            query=dense,
            using=DENSE_VECTOR,
            query_filter=qfilter,
            limit=top_k,
            with_payload=True,
            search_params=search_params,
        )
        results = [
            _payload_to_chunk(hit.payload or {}, float(hit.score or 0.0))
            for hit in response.points
        ]

    # Stable tie-break so equal RRF scores don't shuffle across runs.
    results.sort(key=lambda c: (-(c.score or 0.0), c.chunk_id or ""))
    logger.info("hybrid q=%r top_k=%d hits=%d", query[:80], top_k, len(results))
    return results


def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    regulation_id: str | None = None,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
    collection: str | None = None,
    rewrite: bool = True,
    rewrite_result: Any | None = None,
    do_rerank: bool = True,
    small_to_big: bool = True,
    llm: Any | None = None,
    history: Sequence[Any] | None = None,
    routed: Any | None = None,
    allow_comparison_branch: bool = True,
    hybrid_top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Full Phase-2 pipeline: condense → rewrite → hybrid RRF → rerank → expand.

    - Hybrid candidates: ~30 (``HYBRID_TOP_K``)
    - After rerank: top 5 (``RERANK_TOP_K``)
    - Then small-to-big parent expansion with dedupe
    - Scope/objective queries: prepend configured Scope article chunk(s)

    When ``routed`` (``RoutedQuery``) is provided, hybrid/rerank/budget/multi-reg/
    hard-filter/small-to-big follow that intent's ``PipelineConfig`` — not a
    shared global default.

    Pass ``rewrite_result`` (from a prior ``rewrite_query`` call) with
    ``rewrite=False`` to avoid a second LLM rewrite — that double-call was a
    major source of non-deterministic chunk sets for the same user question.

    ``allow_comparison_branch=False`` disables the dual-corpus comparison path
    (used for per-regulation side retrieves inside that path).

    ``hybrid_top_k`` optionally overrides the hybrid candidate count (used by
    comparison side-retrieves to keep CrossEncoder cost bounded).
    """
    load_dotenv()
    query = (query or "").strip()
    if not query:
        return []

    collection = collection or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    client = client or get_qdrant_client()
    embedder = embedder or Embedder()
    hybrid_k = int(
        hybrid_top_k
        if hybrid_top_k is not None
        else os.getenv("HYBRID_TOP_K", str(DEFAULT_HYBRID_TOP_K))
    )

    from retrieval.context_budget import apply_context_budget, context_budgets
    from retrieval.enumerative import (
        bias_chunks_for_enumerative_topic,
        classify_enumerative,
        resolve_hard_regulation_filter,
        topic_focused_subquery,
    )
    from retrieval.pipelines import overrides_from_routed, record_intent_on_trace

    overrides = overrides_from_routed(routed)
    record_intent_on_trace(routed)
    explicit_regulation_id = bool(regulation_id)

    # Prefer original user text when a rewrite_result is supplied — LLM rewrite
    # often strips "list every" into "what are the requirements…", which would
    # incorrectly drop the enumerative path.
    enum_probe = query
    if rewrite_result is not None:
        enum_probe = (
            str(getattr(rewrite_result, "original", "") or "").strip()
            or str(getattr(rewrite_result, "condensed", "") or "").strip()
            or query
        )
    enum_cls = classify_enumerative(enum_probe)
    if not enum_cls.is_enumerative and enum_probe != query:
        enum_cls = classify_enumerative(query)
    enum = enum_cls.is_enumerative
    if overrides and overrides.force_enumerative:
        enum = True
    budget_q = enum_probe if enum else query
    budget_chunks, _budget_tok, budget_mode = context_budgets(
        budget_q, routed=routed
    )
    logger.info(
        "enumerative_classifier_result fired=%s cue=%r probe=%r query=%r intent=%s",
        enum,
        enum_cls.reason,
        enum_probe[:120],
        query[:120],
        overrides.intent.value if overrides else None,
    )

    # --- Comparison mode: retrieve EACH named regulation separately ----------
    # Ignores a soft regulation_scope lock from eval — pairwise compare needs
    # both corpora. Side retrieves set allow_comparison_branch=False.
    if (
        allow_comparison_branch
        and os.getenv("COMPARISON_RETRIEVE", "1") not in {"0", "false", "False"}
    ):
        from retrieval.comparison import (
            detect_named_regulations,
            is_comparison_mode_query,
            retrieve_comparison,
        )

        probe = enum_probe or query
        if is_comparison_mode_query(probe) or is_comparison_mode_query(query):
            named = detect_named_regulations(probe) or detect_named_regulations(query)
            if len(named) >= 2:
                cmp = retrieve_comparison(
                    probe if is_comparison_mode_query(probe) else query,
                    regulation_ids=named,
                    top_k=top_k or None,
                    llm=llm,
                    client=client,
                    embedder=embedder,
                    collection=collection,
                    do_rerank=do_rerank,
                )
                candidates = list(cmp.chunks)
                candidates, budget_stats = apply_context_budget(
                    candidates, question=probe, routed=routed
                )
                retrieval_log = {
                    "original_query": query,
                    "condensed_for_rerank": probe,
                    "expanded_query": cmp.topic_query,
                    "subqueries": [
                        f"compare:{rid}:{cmp.topic_query}" for rid in cmp.regulation_ids
                    ],
                    "n_hybrid_candidates": len(cmp.chunks),
                    "n_post_rerank": len(candidates),
                    "rerank_ran": bool(do_rerank),
                    "enumerative": enum,
                    "comparison_mode": True,
                    "comparison": cmp.to_public_dict(),
                    "query_intent": overrides.intent.value if overrides else None,
                    "retrieval_strategy": overrides.strategy.value if overrides else None,
                    "final_k": len(candidates),
                    "hybrid_k": hybrid_k,
                    "regulation_id": None,
                    "ignored_explicit_regulation_id": regulation_id
                    if explicit_regulation_id
                    else None,
                    "chunk_ids": [c.chunk_id for c in candidates if c.chunk_id],
                    "context_budget": budget_stats,
                }
                logger.info(
                    "comparison_mode regs=%s chunks=%s missing=%s ignored_scope=%s",
                    cmp.regulation_ids,
                    retrieval_log["chunk_ids"],
                    cmp.missing,
                    retrieval_log["ignored_explicit_regulation_id"],
                )
                try:
                    from observability.context import get_current_trace

                    tr = get_current_trace()
                    if tr is not None:
                        tr.retrieval_queries = list(retrieval_log["subqueries"])
                        tr.retrieval_log = retrieval_log
                        tr.chunk_ids = list(retrieval_log["chunk_ids"])
                        tr.context_chunks_to_llm = int(
                            budget_stats.get("context_chunks_to_llm") or 0
                        )
                        tr.context_tokens_est = int(
                            budget_stats.get("context_tokens_est") or 0
                        )
                        tr.context_budget_mode = str(budget_stats.get("mode") or "")
                        tr.optimizations["comparison_mode"] = True
                        tr.optimizations["comparison"] = cmp.to_public_dict()
                except Exception:  # noqa: BLE001
                    pass
                return candidates

    # Hard metadata filter when the user names a single regulation (UN R94, …).
    # Skip only for comparative / cross-reg asks, or when the intent disables it.
    allow_hard_reg = overrides.hard_reg_filter if overrides else True
    if not regulation_id and allow_hard_reg:
        hard_reg = resolve_hard_regulation_filter(query)
        if hard_reg:
            regulation_id = hard_reg
            logger.info("hard named-regulation filter=%s", regulation_id)

    if overrides:
        hybrid_k = max(hybrid_k, overrides.hybrid_top_k)
        default_final = overrides.rerank_top_k
        # Fix 15: enumerative / limits-aggregation keep broad top-k even when
        # routed as FACTUAL_LOOKUP (which would otherwise crush to top-5).
        if enum:
            hybrid_k = max(hybrid_k, enum_cls.hybrid_top_k)
            default_final = max(default_final, enum_cls.rerank_top_k)
            logger.info(
                "enumerative override — hybrid_k=%d rerank_top_k=%d cue=%r intent=%s",
                hybrid_k,
                default_final,
                enum_cls.reason,
                overrides.intent.value,
            )
        small_to_big = bool(small_to_big and overrides.small_to_big)
        logger.info(
            "routed intent=%s strategy=%s hybrid_k=%d rerank_top_k=%d "
            "multi_reg=%s enum=%s prepend_scope=%s small_to_big=%s budget=%s",
            overrides.intent.value,
            overrides.strategy.value,
            hybrid_k,
            default_final,
            overrides.force_multi_reg,
            enum,
            overrides.prepend_scope,
            small_to_big,
            budget_mode,
        )
    elif enum:
        hybrid_k = max(hybrid_k, enum_cls.hybrid_top_k)
        # Always use the configured enum rerank breadth (e.g. 20), not RERANK_TOP_K=5.
        default_final = enum_cls.rerank_top_k
        logger.info(
            "enumerative path — hybrid_k=%d rerank_top_k=%d context_budget=%d/%s cue=%r",
            hybrid_k,
            default_final,
            budget_chunks,
            budget_mode,
            enum_cls.reason,
        )
    else:
        default_final = int(os.getenv("RERANK_TOP_K", "5"))
    final_k = top_k or default_final
    scope_query = is_scope_objective_query(query)
    definition_query = is_definition_query(query)
    if overrides and overrides.prepend_scope:
        scope_query = True
    article_boost = scope_query or definition_query

    # --- 1) condense (if history) + acronym expand + optional multi-query ---
    from retrieval.acronyms import expand_acronyms
    from retrieval.rewrite import RewriteResult, rewrite_query

    retrieval_query = expand_acronyms(query)
    subqueries = [retrieval_query]
    condensed_for_rerank = query
    rewritten: RewriteResult | None = None

    if rewrite_result is not None:
        rewritten = rewrite_result
        retrieval_query = (
            rewritten.expanded or rewritten.condensed or query  # type: ignore[union-attr]
        )
        condensed_for_rerank = rewritten.condensed or query  # type: ignore[union-attr]
        subqueries = list(rewritten.subqueries or [retrieval_query])  # type: ignore[union-attr]
        if not subqueries:
            subqueries = [retrieval_query]
        logger.info(
            "retrieve reuse rewrite_result original=%r condensed=%r expanded=%r subqueries=%s",
            getattr(rewritten, "original", query),
            condensed_for_rerank,
            retrieval_query,
            subqueries,
        )
    elif rewrite and os.getenv("RETRIEVAL_REWRITE", "1") not in {"0", "false", "False"}:
        rewritten = rewrite_query(query, llm=llm, history=history)
        retrieval_query = rewritten.expanded or rewritten.condensed or query
        condensed_for_rerank = rewritten.condensed or query
        subqueries = rewritten.subqueries or [retrieval_query]
        logger.info(
            "retrieve original=%r condensed=%r expanded=%r subqueries=%s",
            rewritten.original,
            rewritten.condensed,
            rewritten.expanded,
            subqueries,
        )
    elif history:
        from retrieval.rewrite import condense_followup

        condensed, applied = condense_followup(query, list(history), llm=llm)
        retrieval_query = expand_acronyms(condensed)
        condensed_for_rerank = condensed
        subqueries = [retrieval_query]
        logger.info(
            "retrieve condensed-only applied=%s original=%r condensed=%r",
            applied,
            query,
            condensed,
        )
    else:
        logger.info("retrieve acronym-expanded query=%r", retrieval_query)

    # --- 2) hybrid: per-criterion split OR shared subqueries ---------------
    from retrieval.multi_criterion import (
        list_named_criteria,
        merge_per_criterion_chunks,
        per_criterion_top_k,
    )
    from retrieval.value_limit import (
        bias_chunks_for_value_vs_limit,
        criteria_focused_subquery,
        expand_value_vs_limit_query,
        is_compliance_prefer_limit_query,
        is_named_criterion_query,
        is_value_vs_limit_query,
    )

    value_vs_limit = is_value_vs_limit_query(query) or is_value_vs_limit_query(
        condensed_for_rerank
    )
    named_criterion = is_named_criterion_query(
        condensed_for_rerank or query
    ) or is_named_criterion_query(query)
    prefer_limit_bias = (
        value_vs_limit
        or named_criterion
        or is_compliance_prefer_limit_query(query)
        or is_compliance_prefer_limit_query(condensed_for_rerank or "")
    )
    try:
        from retrieval.value_limit import is_definition_seeking_query

        if is_definition_seeking_query(query) or is_definition_seeking_query(
            condensed_for_rerank or ""
        ):
            prefer_limit_bias = False
    except Exception:  # noqa: BLE001
        pass
    multi_criteria = list_named_criteria(condensed_for_rerank or query)
    multi_criterion = len(multi_criteria) >= 2
    crit_sq = criteria_focused_subquery(condensed_for_rerank or query)

    # --- Plural / all-regulation survey: one retrieve per indexed corpus ----
    from retrieval.multi_regulation import (
        is_plural_regulation_query,
        per_regulation_top_k,
        record_multi_regulation_on_trace,
        retrieve_per_indexed_regulation,
    )

    plural_q = is_plural_regulation_query(query) or is_plural_regulation_query(
        condensed_for_rerank or ""
    )
    force_multi = bool(overrides and overrides.force_multi_reg)
    if force_multi and not explicit_regulation_id:
        # Intent requires multi-reg loop — clear a hard-named filter so we survey.
        regulation_id = None
        allow_hard_reg = False

    # Fix 24 safety rail: an explicitly named single regulation always hard-filters
    # at hybrid/dense time. Comparative + plural queries return None from resolve_*
    # and therefore still allow multi-reg / comparative paths.
    if not explicit_regulation_id:
        locked = resolve_hard_regulation_filter(
            condensed_for_rerank or query
        ) or resolve_hard_regulation_filter(query)
        if locked:
            regulation_id = locked
            allow_hard_reg = True
            force_multi = False
            logger.info(
                "fix24 hard regulation lock=%s (blocks cross-reg contamination)",
                locked,
            )

    # --- RETEST_SCOPE: modification / extension-of-approval clauses --------
    if (
        overrides
        and overrides.intent.value == "RETEST_SCOPE"
        and os.getenv("RETEST_SCOPE_RETRIEVE", "1") not in {"0", "false", "False"}
    ):
        from retrieval.retest_scope import expand_retest_query, retrieve_retest_scope

        probe = query
        if rewritten is not None:
            probe = (
                str(getattr(rewritten, "original", "") or "").strip()
                or str(getattr(rewritten, "condensed", "") or "").strip()
                or query
            )
        expansion = expand_retest_query(probe)
        if expansion.named_regulation_id and not explicit_regulation_id:
            regulation_id = expansion.named_regulation_id
        retest = retrieve_retest_scope(
            condensed_for_rerank or query,
            client=client,
            embedder=embedder,
            collection=collection,
            expansion=expansion,
        )
        candidates = list(retest.chunks)
        candidates, budget_stats = apply_context_budget(
            candidates, question=condensed_for_rerank or query, routed=routed
        )
        retrieval_log = {
            "original_query": query,
            "condensed_for_rerank": condensed_for_rerank,
            "expanded_query": retrieval_query,
            "subqueries": [
                f"modification:{rid}" for rid in expansion.target_regulation_ids
            ],
            "n_hybrid_candidates": len(retest.chunks),
            "n_post_rerank": len(candidates),
            "rerank_ran": False,
            "enumerative": enum,
            "retest_scope": True,
            "retest_retrieval": retest.to_public_dict(),
            "query_intent": overrides.intent.value,
            "retrieval_strategy": overrides.strategy.value,
            "final_k": len(candidates),
            "hybrid_k": hybrid_k,
            "regulation_id": expansion.named_regulation_id,
            "chunk_ids": [c.chunk_id for c in candidates if c.chunk_id],
            "context_budget": budget_stats,
        }
        logger.info(
            "retest_scope targets=%s changes=%s chunks=%s mod_lang=%s",
            expansion.target_regulation_ids,
            [c.id for c in expansion.changes],
            retrieval_log["chunk_ids"],
            retest.found_modification_language,
        )
        try:
            from observability.context import get_current_trace

            tr = get_current_trace()
            if tr is not None:
                tr.retrieval_queries = list(retrieval_log["subqueries"])
                tr.retrieval_log = retrieval_log
                tr.chunk_ids = list(retrieval_log["chunk_ids"])
                tr.context_chunks_to_llm = int(
                    budget_stats.get("context_chunks_to_llm") or 0
                )
                tr.context_tokens_est = int(budget_stats.get("context_tokens_est") or 0)
                tr.context_budget_mode = str(budget_stats.get("mode") or "")
                tr.optimizations["retest_scope"] = True
                tr.optimizations["retest_retrieval"] = retest.to_public_dict()
                from retrieval.retest_scope import (
                    RETEST_DISCLAIMER,
                    RETEST_DISCLAIMER_TITLE,
                )

                tr.optimizations["mode_disclaimer"] = RETEST_DISCLAIMER
                tr.optimizations["mode_disclaimer_title"] = RETEST_DISCLAIMER_TITLE
        except Exception:  # noqa: BLE001
            pass
        return candidates

    # --- APPLICABILITY: Scope clause of EACH indexed regulation -------------
    # Fix 24: skip when locked to a single named regulation_id.
    if (
        overrides
        and overrides.intent.value == "APPLICABILITY"
        and not regulation_id
        and os.getenv("APPLICABILITY_RETRIEVE", "1") not in {"0", "false", "False"}
    ):
        from retrieval.applicability import (
            expand_applicability_query,
            retrieve_applicability,
        )

        probe = query
        if rewritten is not None:
            probe = (
                str(getattr(rewritten, "original", "") or "").strip()
                or str(getattr(rewritten, "condensed", "") or "").strip()
                or query
            )
        expansion = expand_applicability_query(probe)
        app = retrieve_applicability(
            condensed_for_rerank or query,
            client=client,
            collection=collection,
            expansion=expansion,
        )
        candidates = list(app.chunks)
        candidates, budget_stats = apply_context_budget(
            candidates, question=condensed_for_rerank or query, routed=routed
        )
        retrieval_log = {
            "original_query": query,
            "condensed_for_rerank": condensed_for_rerank,
            "expanded_query": retrieval_query,
            "subqueries": [
                f"scope:{rid}" for rid in expansion.indexed_regulation_ids
            ],
            "n_hybrid_candidates": len(app.chunks),
            "n_post_rerank": len(candidates),
            "rerank_ran": False,
            "enumerative": enum,
            "applicability": True,
            "multi_regulation": True,
            "applicability_retrieval": app.to_public_dict(),
            "multi_regulation_covered": sorted(app.per_regulation.keys()),
            "multi_regulation_missing": list(app.missing_scope),
            "query_intent": overrides.intent.value,
            "retrieval_strategy": overrides.strategy.value,
            "final_k": len(candidates),
            "hybrid_k": hybrid_k,
            "regulation_id": None,
            "chunk_ids": [c.chunk_id for c in candidates if c.chunk_id],
            "context_budget": budget_stats,
        }
        logger.info(
            "applicability scopes=%s missing=%s vehicle=%s",
            sorted(app.per_regulation.keys()),
            app.missing_scope,
            expansion.vehicle.to_public_dict(),
        )
        try:
            from observability.context import get_current_trace

            tr = get_current_trace()
            if tr is not None:
                tr.retrieval_queries = list(retrieval_log["subqueries"])
                tr.retrieval_log = retrieval_log
                tr.chunk_ids = list(retrieval_log["chunk_ids"])
                tr.context_chunks_to_llm = int(
                    budget_stats.get("context_chunks_to_llm") or 0
                )
                tr.context_tokens_est = int(budget_stats.get("context_tokens_est") or 0)
                tr.context_budget_mode = str(budget_stats.get("mode") or "")
                tr.optimizations["applicability"] = True
                tr.optimizations["applicability_retrieval"] = app.to_public_dict()
        except Exception:  # noqa: BLE001
            pass
        return candidates

    # --- SCOPE_SUMMARY: hard-reg + deterministic section fetch -------------
    if (
        overrides
        and overrides.intent.value == "SCOPE_SUMMARY"
        and os.getenv("SCOPE_SUMMARY_RETRIEVE", "1") not in {"0", "false", "False"}
    ):
        from retrieval.scope_summary import (
            expand_scope_summary_query,
            retrieve_scope_summary,
        )

        probe = query
        if rewritten is not None:
            probe = (
                str(getattr(rewritten, "original", "") or "").strip()
                or str(getattr(rewritten, "condensed", "") or "").strip()
                or query
            )
        # Prefer routed.regulation_id / explicit arg — never leave unnamed.
        named = (
            regulation_id
            or (getattr(routed, "regulation_id", None) if routed is not None else None)
            or None
        )
        expansion = expand_scope_summary_query(probe, regulation_id=named)
        if not expansion.regulation_id:
            logger.warning(
                "scope_summary: no named regulation for %r — refusing cross-reg retrieve",
                probe[:120],
            )
            return []
        regulation_id = expansion.regulation_id
        scope_res = retrieve_scope_summary(
            condensed_for_rerank or query,
            regulation_id=regulation_id,
            client=client,
            collection=collection,
            expansion=expansion,
        )
        candidates = list(scope_res.chunks)
        candidates, budget_stats = apply_context_budget(
            candidates, question=condensed_for_rerank or query, routed=routed
        )
        # Post-budget hard filter — never leak foreign regs.
        candidates = [
            c
            for c in candidates
            if (c.regulation_id or "").strip() == regulation_id
        ]
        retrieval_log = {
            "original_query": query,
            "condensed_for_rerank": condensed_for_rerank,
            "expanded_query": retrieval_query,
            "subqueries": [f"scope_summary:{regulation_id}"],
            "n_hybrid_candidates": len(scope_res.chunks),
            "n_post_rerank": len(candidates),
            "rerank_ran": False,
            "enumerative": enum,
            "scope_summary": True,
            "scope_summary_retrieval": scope_res.to_public_dict(),
            "query_intent": overrides.intent.value,
            "retrieval_strategy": overrides.strategy.value,
            "final_k": len(candidates),
            "hybrid_k": hybrid_k,
            "regulation_id": regulation_id,
            "chunk_ids": [c.chunk_id for c in candidates if c.chunk_id],
            "context_budget": budget_stats,
        }
        logger.info(
            "scope_summary reg=%s chunks=%s roles=%s",
            regulation_id,
            retrieval_log["chunk_ids"],
            scope_res.to_public_dict().get("roles"),
        )
        try:
            from observability.context import get_current_trace

            tr = get_current_trace()
            if tr is not None:
                tr.retrieval_queries = list(retrieval_log["subqueries"])
                tr.retrieval_log = retrieval_log
                tr.chunk_ids = list(retrieval_log["chunk_ids"])
                tr.context_chunks_to_llm = int(
                    budget_stats.get("context_chunks_to_llm") or 0
                )
                tr.context_tokens_est = int(budget_stats.get("context_tokens_est") or 0)
                tr.context_budget_mode = str(budget_stats.get("mode") or "")
                tr.optimizations["scope_summary"] = True
                tr.optimizations["scope_summary_retrieval"] = scope_res.to_public_dict()
        except Exception:  # noqa: BLE001
            pass
        return candidates

    # --- CHECKLIST_GEN: per-category separate retrievals --------------------
    if (
        overrides
        and overrides.intent.value == "CHECKLIST_GEN"
        and os.getenv("CHECKLIST_GEN_RETRIEVE", "1") not in {"0", "false", "False"}
    ):
        from retrieval.checklist import (
            expand_checklist_query,
            is_checklist_pipeline_query,
            retrieve_checklist,
        )

        probe = query
        if rewritten is not None:
            probe = (
                str(getattr(rewritten, "original", "") or "").strip()
                or str(getattr(rewritten, "condensed", "") or "").strip()
                or query
            )
        # Homologation/prep checklist only — bare "list every…" stays enumerative.
        if is_checklist_pipeline_query(probe) or is_checklist_pipeline_query(query):
            expansion = expand_checklist_query(probe)
            if expansion.regulation_id and not explicit_regulation_id:
                regulation_id = expansion.regulation_id
            check = retrieve_checklist(
                condensed_for_rerank or query,
                top_k_per_category=top_k or 4,
                client=client,
                embedder=embedder,
                collection=collection,
                do_rerank=do_rerank,
                expansion=expansion,
            )
            candidates = list(check.chunks)
            candidates, budget_stats = apply_context_budget(
                candidates, question=condensed_for_rerank or query, routed=routed
            )
            retrieval_log = {
                "original_query": query,
                "condensed_for_rerank": condensed_for_rerank,
                "expanded_query": retrieval_query,
                "subqueries": [
                    q
                    for row in check.by_category
                    for q in (row.category.retrieval_queries[:1] or [row.category.label])
                ],
                "n_hybrid_candidates": len(check.chunks),
                "n_post_rerank": len(candidates),
                "rerank_ran": bool(do_rerank),
                "enumerative": enum,
                "checklist_gen": True,
                "checklist_retrieval": check.to_public_dict(),
                "query_intent": overrides.intent.value,
                "retrieval_strategy": overrides.strategy.value,
                "final_k": top_k or 4,
                "hybrid_k": hybrid_k,
                "regulation_id": expansion.regulation_id,
                "chunk_ids": [c.chunk_id for c in candidates if c.chunk_id],
                "context_budget": budget_stats,
            }
            logger.info(
                "checklist_gen reg=%s covered=%s missing=%s chunks=%s",
                expansion.regulation_id,
                check.covered_categories,
                check.missing_categories,
                retrieval_log["chunk_ids"],
            )
            try:
                from observability.context import get_current_trace

                tr = get_current_trace()
                if tr is not None:
                    tr.retrieval_queries = list(retrieval_log["subqueries"])
                    tr.retrieval_log = retrieval_log
                    tr.chunk_ids = list(retrieval_log["chunk_ids"])
                    tr.context_chunks_to_llm = int(
                        budget_stats.get("context_chunks_to_llm") or 0
                    )
                    tr.context_tokens_est = int(budget_stats.get("context_tokens_est") or 0)
                    tr.context_budget_mode = str(budget_stats.get("mode") or "")
                    tr.optimizations["checklist_gen"] = True
                    tr.optimizations["checklist_retrieval"] = check.to_public_dict()
            except Exception:  # noqa: BLE001
                pass
            return candidates

    # --- DESIGN_IMPLICATION: concept expansion + per-reg retrieve -----------
    # Fix 24: named-reg design asks stay inside that regulation (standard path).
    if (
        overrides
        and overrides.intent.value == "DESIGN_IMPLICATION"
        and not regulation_id
        and os.getenv("DESIGN_IMPLICATION_RETRIEVE", "1") not in {"0", "false", "False"}
    ):
        from retrieval.design_implication import (
            expand_design_query,
            per_regulation_design_top_k,
            retrieve_design_implication,
        )

        probe = query
        if rewritten is not None:
            probe = (
                str(getattr(rewritten, "original", "") or "").strip()
                or str(getattr(rewritten, "condensed", "") or "").strip()
                or query
            )
        expansion = expand_design_query(probe)
        # Named regulation in the question (e.g. UN R16) → single-corpus design path.
        if expansion.named_regulation_id and not explicit_regulation_id:
            regulation_id = expansion.named_regulation_id
        # Fix 13: per-reg cap stays small; pipeline max_chunks trims the merge.
        per_reg_k = top_k or per_regulation_design_top_k()
        if overrides and getattr(overrides, "rerank_top_k", None):
            per_reg_k = min(per_reg_k, max(2, int(overrides.rerank_top_k) // 2))
        design = retrieve_design_implication(
            condensed_for_rerank or query,
            top_k_per_reg=per_reg_k,
            llm=llm,
            rewrite_result=rewritten,
            do_rerank=do_rerank,
            client=client,
            embedder=embedder,
            collection=collection,
            expansion=expansion,
        )
        candidates = list(design.chunks)
        candidates, budget_stats = apply_context_budget(
            candidates, question=condensed_for_rerank or query, routed=routed
        )
        from retrieval.design_implication import per_regulation_design_top_k as _design_k

        retrieval_log = {
            "original_query": query,
            "condensed_for_rerank": condensed_for_rerank,
            "expanded_query": retrieval_query,
            "subqueries": list(design.expansion.subqueries),
            "n_hybrid_candidates": len(design.chunks),
            "n_post_rerank": len(candidates),
            "rerank_ran": bool(do_rerank),
            "enumerative": enum,
            "multi_regulation": True,
            "design_implication": True,
            "design_expansion": design.expansion.to_public_dict(),
            "multi_regulation_covered": list(design.covered),
            "multi_regulation_missing": list(design.missing),
            "query_intent": overrides.intent.value,
            "retrieval_strategy": overrides.strategy.value,
            "final_k": top_k or _design_k(),
            "hybrid_k": hybrid_k,
            "regulation_id": expansion.named_regulation_id,
            "chunk_ids": [c.chunk_id for c in candidates if c.chunk_id],
            "context_budget": budget_stats,
        }
        logger.info(
            "design_implication survey component=%s covered=%s missing=%s chunks=%s",
            design.expansion.component.id if design.expansion.component else None,
            design.covered,
            design.missing,
            retrieval_log["chunk_ids"],
        )
        try:
            from observability.context import get_current_trace

            tr = get_current_trace()
            if tr is not None:
                tr.retrieval_queries = list(design.expansion.subqueries)
                tr.retrieval_log = retrieval_log
                tr.chunk_ids = list(retrieval_log["chunk_ids"])
                tr.context_chunks_to_llm = int(budget_stats.get("context_chunks_to_llm") or 0)
                tr.context_tokens_est = int(budget_stats.get("context_tokens_est") or 0)
                tr.context_budget_mode = str(budget_stats.get("mode") or "")
                tr.optimizations["design_implication"] = True
                tr.optimizations["design_expansion"] = design.expansion.to_public_dict()
        except Exception:  # noqa: BLE001
            pass
        return candidates

    if (
        not regulation_id
        and (plural_q or force_multi)
        and os.getenv("MULTI_REG_RETRIEVE", "1") not in {"0", "false", "False"}
    ):
        per_k = top_k or (
            overrides.rerank_top_k if (overrides and force_multi) else per_regulation_top_k()
        )
        # Per-reg k for design/applicability: keep modest per corpus, rely on merge.
        if force_multi:
            per_k = min(per_k, per_regulation_top_k() + 2)
        multi = retrieve_per_indexed_regulation(
            condensed_for_rerank or query,
            top_k=per_k,
            llm=llm,
            rewrite=False,
            rewrite_result=rewritten,
            do_rerank=do_rerank,
            small_to_big=False,
            client=client,
            embedder=embedder,
            collection=collection,
        )
        record_multi_regulation_on_trace(multi)
        candidates = list(multi.chunks)
        candidates, budget_stats = apply_context_budget(
            candidates, question=condensed_for_rerank or query, routed=routed
        )
        retrieval_log = {
            "original_query": query,
            "condensed_for_rerank": condensed_for_rerank,
            "expanded_query": retrieval_query,
            "subqueries": list(subqueries),
            "n_hybrid_candidates": len(multi.chunks),
            "n_post_rerank": len(candidates),
            "rerank_ran": bool(do_rerank),
            "enumerative": enum,
            "multi_regulation": True,
            "multi_regulation_covered": list(multi.covered),
            "multi_regulation_missing": list(multi.missing),
            "query_intent": overrides.intent.value if overrides else None,
            "retrieval_strategy": overrides.strategy.value if overrides else None,
            "final_k": per_k,
            "hybrid_k": hybrid_k,
            "regulation_id": None,
            "chunk_ids": [c.chunk_id for c in candidates if c.chunk_id],
            "context_budget": budget_stats,
        }
        logger.info(
            "multi-regulation survey covered=%s missing=%s chunks=%s intent=%s",
            multi.covered,
            multi.missing,
            retrieval_log["chunk_ids"],
            overrides.intent.value if overrides else None,
        )
        try:
            from observability.context import get_current_trace

            tr = get_current_trace()
            if tr is not None:
                tr.retrieval_queries = list(subqueries)
                tr.retrieval_log = retrieval_log
                tr.chunk_ids = list(retrieval_log["chunk_ids"])
                tr.context_chunks_to_llm = int(budget_stats.get("context_chunks_to_llm") or 0)
                tr.context_tokens_est = int(budget_stats.get("context_tokens_est") or 0)
                tr.context_budget_mode = str(budget_stats.get("mode") or "")
        except Exception:  # noqa: BLE001
            pass
        return candidates

    # Re-resolve after condensation (follow-ups may only name the reg in history).
    if not regulation_id and allow_hard_reg:
        hard_reg = resolve_hard_regulation_filter(condensed_for_rerank or query)
        if hard_reg:
            regulation_id = hard_reg
            logger.info(
                "hard named-regulation filter (post-condense)=%s",
                regulation_id,
            )

    per_k = per_criterion_top_k()
    if multi_criterion:
        # Separate hybrid call per criterion so fuel leakage (etc.) cannot lose
        # a shared top-k fight against HPC / ThCC in one combined query.
        per_lists: list[list[RetrievedChunk]] = []
        for crit in multi_criteria:
            sq = crit.retrieval_query
            # Keep regulation / pass-fail cues from the user question when present.
            if condensed_for_rerank and condensed_for_rerank not in sq:
                sq = f"{crit.matched_text} {crit.retrieval_query}"
            hits = hybrid_search(
                sq,
                top_k=max(hybrid_k, per_k),
                regulation_id=regulation_id,
                client=client,
                embedder=embedder,
                collection=collection,
            )
            if prefer_limit_bias:
                hits = bias_chunks_for_value_vs_limit(
                    hits, question=f"{crit.matched_text} {condensed_for_rerank or query}"
                )
            top = hits[:per_k]
            per_lists.append(top)
            logger.info(
                "multi-criterion retrieve key=%s matched=%r top=%s",
                crit.key,
                crit.matched_text,
                [(c.chunk_id[:12], c.section_number) for c in top],
            )
        candidates = merge_per_criterion_chunks(per_lists, per_k=per_k)
        # Keep enough headroom for every criterion after rerank.
        final_k = max(final_k, per_k * len(multi_criteria))
        subqueries = [
            f"{c.matched_text} {c.retrieval_query}" for c in multi_criteria
        ]
        logger.info(
            "multi-criterion merge n_criteria=%d per_k=%d merged=%d final_k=%d",
            len(multi_criteria),
            per_k,
            len(candidates),
            final_k,
        )
    else:
        if value_vs_limit:
            # Bias BM25/dense toward injury-criteria limits, not sensor calibration.
            subqueries = [expand_value_vs_limit_query(s) for s in subqueries]

        # Always add a criteria-focused subquery for named injury metrics (HPC/RDC/…),
        # not only for measured value-vs-limit pass/fail questions.
        if crit_sq and crit_sq not in subqueries:
            subqueries.append(crit_sq)
            logger.info(
                "named-criterion subquery added crit_sq=%r value_vs_limit=%s",
                crit_sq,
                value_vs_limit,
            )

        if value_vs_limit:
            logger.info(
                "value_vs_limit query — criteria-biased subqueries=%s",
                subqueries,
            )

        # Stable subquery order → stable RRF when the set is identical.
        subqueries = list(dict.fromkeys(s.strip() for s in subqueries if (s or "").strip()))
        # Enumerative + topic (doors, …): keep a focused keyword subquery so
        # scattered requirement clauses beat preamble/admin dense neighbors.
        if enum:
            topic_sq = topic_focused_subquery(enum_probe or condensed_for_rerank or query)
            if topic_sq and topic_sq not in subqueries:
                subqueries.append(topic_sq)
                # Also keep the original list-every phrasing if rewrite replaced it.
                if enum_probe and enum_probe not in subqueries:
                    subqueries.insert(0, expand_acronyms(enum_probe))
                logger.info("enumerative topic subquery=%r subqueries=%s", topic_sq, subqueries)

        logger.info(
            "retrieve hybrid_subqueries=%s expanded_primary=%r",
            subqueries,
            retrieval_query,
        )

        lists: list[list[RetrievedChunk]] = []
        for sq in subqueries:
            lists.append(
                hybrid_search(
                    sq,
                    top_k=hybrid_k,
                    regulation_id=regulation_id,
                    client=client,
                    embedder=embedder,
                    collection=collection,
                )
            )
        candidates = (
            rrf_merge(lists, top_k=hybrid_k) if len(lists) > 1 else (lists[0] if lists else [])
        )
        if prefer_limit_bias and candidates:
            # Prefer performance-criteria clauses before cross-encoder / top-k cut.
            candidates = bias_chunks_for_value_vs_limit(
                candidates, question=condensed_for_rerank or query
            )

    n_hybrid = len(candidates)
    n_post_rerank = 0

    # --- 3) cross-encoder rerank ------------------------------------------
    # Multi-criterion: rerank within a pool large enough to keep ≥per_k×n, but
    # do not cut below the merged set size so no criterion is dropped.
    rerank_ran = False
    if candidates:
        if do_rerank and os.getenv("RETRIEVAL_RERANK", "1") not in {"0", "false", "False"}:
            from retrieval.rerank import rerank

            # Score against the condensed standalone question.
            rerank_n = final_k if not multi_criterion else max(final_k, len(candidates))
            candidates = rerank(condensed_for_rerank, candidates, top_n=rerank_n)
            rerank_ran = True
            if multi_criterion:
                # Re-merge quota: ensure each criterion still has a covering chunk
                # when possible, then pad with rerank order.
                from retrieval.multi_criterion import criterion_covered_by_chunks

                kept: list[RetrievedChunk] = []
                seen_ids: set[str] = set()
                for crit in multi_criteria:
                    for c in candidates:
                        cid = c.chunk_id or ""
                        if cid in seen_ids:
                            continue
                        if criterion_covered_by_chunks(crit, [c]):
                            kept.append(c)
                            seen_ids.add(cid)
                            break
                for c in candidates:
                    cid = c.chunk_id or ""
                    if cid in seen_ids:
                        continue
                    kept.append(c)
                    seen_ids.add(cid)
                    if len(kept) >= final_k:
                        break
                candidates = kept[:final_k]
            elif prefer_limit_bias:
                # Re-apply criteria bias after rerank so ISO 6487 / electrical
                # cannot win over an explicit HPC/RDC limit clause.
                candidates = bias_chunks_for_value_vs_limit(
                    candidates, question=condensed_for_rerank or query
                )[:final_k]
            elif enum:
                candidates = bias_chunks_for_enumerative_topic(
                    candidates, question=enum_probe or condensed_for_rerank or query
                )[:final_k]
            else:
                candidates = candidates[:final_k]
        else:
            candidates = candidates[:final_k]
        if enum and candidates:
            # Ensure topic bias even when rerank was skipped.
            candidates = bias_chunks_for_enumerative_topic(
                candidates, question=enum_probe or condensed_for_rerank or query
            )[:final_k]

        # Section-category soft boost (requirements / installation / scope / definitions).
        try:
            from retrieval.limits_aggregation import is_limits_aggregation_query
            from retrieval.section_bias import (
                bias_chunks_by_section_category,
                detect_section_category,
            )

            sec_cat = detect_section_category(condensed_for_rerank or query)
            if sec_cat and candidates and not is_limits_aggregation_query(
                condensed_for_rerank or query
            ):
                candidates = bias_chunks_by_section_category(
                    candidates, condensed_for_rerank or query, category=sec_cat
                )[:final_k]
        except Exception as exc:  # noqa: BLE001
            logger.debug("section_category_bias skipped: %s", exc)

        n_post_rerank = len(candidates)
        logger.info(
            "rerank_cut hybrid=%d → post_rerank=%d final_k=%d rerank_ran=%s multi=%s",
            n_hybrid,
            n_post_rerank,
            final_k,
            rerank_ran,
            multi_criterion,
        )

        # --- 4) small-to-big parent expansion -----------------------------
        # Skip for scope/definition article boosts so the clause stays clean.
        # Skip for value-vs-limit / named-criterion so injury-criteria §5 is not
        # merged into a huge Specifications blob (or Annex procedure text).
        # Oversized parents (whole Annex) are rejected inside expand_to_parents.
        if (
            small_to_big
            and not article_boost
            and not value_vs_limit
            and not named_criterion
            and not multi_criterion
            and not enum  # keep leaf door/requirement clauses; parents drown them
            and os.getenv("RETRIEVAL_SMALL_TO_BIG", "1") not in {"0", "false", "False"}
        ):
            from retrieval.expand import expand_to_parents

            candidates = expand_to_parents(
                candidates, client=client, collection=collection
            )

    # --- 5) article boost: Scope / Definitions ahead of hybrid ------------
    if scope_query:
        leading = fetch_scope_chunks(
            regulation_id=regulation_id,
            client=client,
            collection=collection,
        )
        lead_ids = {c.chunk_id for c in leading if c.chunk_id}
        hybrid_tail = [c for c in candidates if c.chunk_id not in lead_ids][:1]
        candidates = prepend_unique_chunks(leading, hybrid_tail)
        logger.info(
            "scope query detected — prepended %d scope chunk(s), hybrid_tail=%d",
            len(leading),
            len(hybrid_tail),
        )
    elif definition_query:
        leading = fetch_definitions_chunks(
            regulation_id=regulation_id,
            client=client,
            collection=collection,
        )
        lead_ids = {c.chunk_id for c in leading if c.chunk_id}
        hybrid_tail = [c for c in candidates if c.chunk_id not in lead_ids][:2]
        candidates = prepend_unique_chunks(leading, hybrid_tail)
        logger.info(
            "definition query detected — prepended %d definitions chunk(s), hybrid_tail=%d",
            len(leading),
            len(hybrid_tail),
        )

    # --- 6) hard context budget before anything leaves retrieve() ---------
    budget_q = condensed_for_rerank or query
    candidates, budget_stats = apply_context_budget(
        candidates, question=budget_q, routed=routed
    )

    # Structured retrieval log (trace + logger) for determinism / cost debugging.
    retrieval_log = {
        "original_query": query,
        "condensed_for_rerank": condensed_for_rerank,
        "expanded_query": retrieval_query,
        "subqueries": list(subqueries),
        "n_hybrid_candidates": n_hybrid,
        "n_post_rerank": n_post_rerank,
        "rerank_ran": rerank_ran,
        "enumerative": enum,
        "enumerative_cue": enum_cls.reason if enum else "",
        "query_intent": overrides.intent.value if overrides else None,
        "retrieval_strategy": overrides.strategy.value if overrides else None,
        "final_k": final_k,
        "hybrid_k": hybrid_k,
        "regulation_id": regulation_id,
        "multi_criterion": multi_criterion,
        "multi_criterion_keys": [c.key for c in multi_criteria] if multi_criterion else [],
        "chunk_ids": [c.chunk_id for c in candidates if c.chunk_id],
        "chunk_scores": [
            {"chunk_id": c.chunk_id, "score": round(float(c.score or 0.0), 6)}
            for c in candidates
            if c.chunk_id
        ],
        "context_budget": budget_stats,
    }
    logger.info(
        "retrieval_final_queries=%s chunk_ids=%s context_chunks_to_llm=%s tokens≈%s",
        retrieval_log["subqueries"],
        retrieval_log["chunk_ids"],
        budget_stats.get("context_chunks_to_llm"),
        budget_stats.get("context_tokens_est"),
    )
    try:
        from observability.context import get_current_trace

        tr = get_current_trace()
        if tr is not None:
            tr.retrieval_queries = list(subqueries)
            tr.retrieval_log = retrieval_log
            tr.chunk_ids = list(retrieval_log["chunk_ids"])
            tr.context_chunks_to_llm = int(budget_stats.get("context_chunks_to_llm") or 0)
            tr.context_tokens_est = int(budget_stats.get("context_tokens_est") or 0)
            tr.context_budget_mode = str(budget_stats.get("mode") or "")
            tr.n_hybrid_candidates = int(n_hybrid)
            tr.rerank_ran = bool(rerank_ran)
            if multi_criterion:
                tr.optimizations["multi_criterion"] = True
                tr.optimizations["multi_criterion_keys"] = [
                    c.key for c in multi_criteria
                ]
                tr.optimizations["multi_criterion_top_k"] = per_k
            if enum:
                tr.optimizations["enumerative"] = True
                tr.optimizations["enumerative_rerank_top_k"] = final_k
                if regulation_id:
                    tr.optimizations["enumerative_regulation_id"] = regulation_id
    except Exception:  # noqa: BLE001
        pass

    logger.info("retrieve pipeline final=%d", len(candidates))
    return candidates


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Join retrieved chunks into the LLM context block."""
    return "\n\n".join(c.context_block(index=i + 1) for i, c in enumerate(chunks))
