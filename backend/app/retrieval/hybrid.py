"""
Hybrid retriever: dense (Nomic) + BM25 with Reciprocal Rank Fusion.
Vectorized cosine search over precomputed embeddings.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from backend.app.core.settings import (
    BM25_WEIGHT,
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_TRUST_REMOTE_CODE,
    ENABLE_METADATA_FILTER,
    ENABLE_MULTI_QUERY,
    ENABLE_PARENT_CHILD,
    ENABLE_QUERY_EXPANSION,
    METADATA_BOOST,
    RRF_K,
    SEMANTIC_WEIGHT,
    TOP_K_CHUNKS,
    TOP_K_RETRIEVE,
    TOP_K_VECTOR,
    VECTOR_SCORE_THRESHOLD,
)
from backend.app.retrieval.query_expansion import (
    expand_query,
    generate_multi_queries,
)

REG_MAP = {
    "un r14": "UN_R14",
    "un r16": "UN_R16",
    "un r17": "UN_R17",
    "un r94": "UN_R94",
    "un r95": "UN_R95",
    "un r137": "UN_R137",
    "fmvss": "FMVSS",
}

_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def _load_json_artifact(path: Path, *, label: str) -> dict:
    """Load a JSON artifact; fail clearly if Git LFS was not pulled."""
    raw = path.read_text(encoding="utf-8")
    if raw.lstrip().startswith(_LFS_POINTER_PREFIX):
        raise ValueError(
            f"{label} ({path.name}) is a Git LFS pointer, not the real file. "
            "Enable Git LFS on Railway or rebuild the Docker image (Dockerfile "
            "runs git lfs pull for output/regulation_embeddings.json)."
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} ({path.name}) is not valid JSON: {exc}") from exc


class HybridRetriever:
    def __init__(
        self,
        *,
        chunks_file: Path | None = None,
        embeddings_file: Path | None = None,
    ) -> None:
        self._chunks_file = chunks_file or CHUNKS_FILE
        self._embeddings_file = embeddings_file or EMBEDDINGS_FILE
        self.chunks: list[dict] = []
        self.embeddings: dict[str, list[float]] = {}
        self._chunk_by_id: dict[str, dict] = {}
        self._model = None
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[dict] = []
        self._emb_matrix: np.ndarray | None = None
        self._emb_norms: np.ndarray | None = None
        self._matrix_chunk_ids: list[str] = []
        self._semantic_disabled = (
            os.getenv("DISABLE_SEMANTIC", "false").lower() == "true"
        )
        self._load()

    def _load(self) -> None:
        if self._chunks_file.exists():
            data = _load_json_artifact(self._chunks_file, label="Chunk index")
            self.chunks = data.get("chunks", [])
            self._chunk_by_id = {
                c.get("chunk_id", ""): c for c in self.chunks if c.get("chunk_id")
            }
            logger.info(f"Loaded {len(self.chunks)} chunks")

        if self._embeddings_file.exists():
            try:
                data = _load_json_artifact(self._embeddings_file, label="Embedding index")
                self.embeddings = data.get("embeddings", {})
                logger.info(f"Loaded {len(self.embeddings)} embeddings")
            except ValueError as exc:
                logger.error(f"{exc} — semantic search disabled; BM25-only fallback")

        self._build_bm25_index()
        self._build_vector_index()

    def _build_bm25_index(self) -> None:
        tokenized = []
        valid = []
        for c in self.chunks:
            txt = (
                f"{c.get('heading_path', '')} "
                f"{c.get('section_title', '')} "
                f"{c.get('text', '')}"
            )
            toks = re.sub(r"[^a-z0-9]", " ", txt.lower()).split()
            if toks:
                tokenized.append(toks)
                valid.append(c)
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_chunks = valid

    def _build_vector_index(self) -> None:
        vectors: list[list[float]] = []
        ids: list[str] = []
        for cid, emb in self.embeddings.items():
            if cid in self._chunk_by_id and emb:
                vectors.append(emb)
                ids.append(cid)
        if not vectors:
            return
        mat = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)
        self._emb_matrix = mat
        self._emb_norms = mat / norms
        self._matrix_chunk_ids = ids
        logger.info(f"Vector index ready: {len(ids)} vectors")

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model (first query may take ~30-60s): {EMBEDDING_MODEL}")
                self._model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE,
                )
                logger.info("Embedding model ready")
            except Exception as exc:
                logger.error(f"Embedding model failed: {exc}")
                raise
        return self._model

    def warmup(self) -> None:
        """Load embedding model and run one encode (avoids delay on first chat)."""
        if self._emb_norms is None:
            logger.warning("No vector index — check output/regulation_embeddings.json")
            return
        model = self._get_model()
        model.encode(["warmup"], convert_to_numpy=True, show_progress_bar=False)
        logger.info("Embedding model warmed up")

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a single string with the shared embedding model.

        Exposed so the gateway's semantic cache can REUSE this exact embedding
        model (nomic-embed-text-v1.5) rather than introducing a second one.
        Uses the query prefix so cached prompts share the query embedding space.
        """
        model = self._get_model()
        return model.encode(
            [EMBEDDING_QUERY_PREFIX + text], convert_to_numpy=True, show_progress_bar=False
        )[0]

    def detect_regs(self, query: str) -> list[str]:
        """Public wrapper around regulation detection (used for cache scoping)."""
        return self._detect_regs(query)

    def _detect_regs(self, query: str) -> list[str]:
        q = query.lower()
        return [v for k, v in REG_MAP.items() if k in q]

    def _filter_chunk_ids(self, regs: list[str]) -> set[str] | None:
        if not regs:
            return None
        return {
            cid
            for cid, c in self._chunk_by_id.items()
            if c.get("regulation", "") in regs
        }

    def _semantic_search(self, query: str, allowed_ids: set[str] | None) -> list[dict]:
        if (
            self._semantic_disabled
            or self._emb_norms is None
            or not self._matrix_chunk_ids
        ):
            return []

        try:
            model = self._get_model()
            # Nomic requires the "search_query: " task prefix on queries.
            q = model.encode(
                [EMBEDDING_QUERY_PREFIX + query],
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            qn = q / max(float(np.linalg.norm(q)), 1e-9)
            sims = self._emb_norms @ qn

            if allowed_ids is not None:
                mask = np.array(
                    [cid in allowed_ids for cid in self._matrix_chunk_ids],
                    dtype=bool,
                )
                sims = np.where(mask, sims, -1.0)

            top_idx = np.argpartition(sims, -TOP_K_VECTOR)[-TOP_K_VECTOR :]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

            results = []
            for i in top_idx:
                sim = float(sims[i])
                if sim < VECTOR_SCORE_THRESHOLD:
                    continue
                cid = self._matrix_chunk_ids[i]
                c = self._chunk_by_id[cid]
                results.append(
                    {
                        "id": cid,
                        "score": sim,
                        # Preserve raw cosine so it survives RRF/multi-query fusion
                        # and can be used for the grounding confidence gate.
                        "semantic_score": sim,
                        "text": c.get("text", ""),
                        "title": c.get("section_title", ""),
                        "heading_path": c.get("heading_path", ""),
                        "regulation": c.get("regulation", ""),
                        "chunk_type": c.get("chunk_type", ""),
                        "parent_id": c.get("parent_id"),
                        "source": "semantic",
                    }
                )
            return results[:TOP_K_VECTOR]
        except Exception as exc:
            logger.warning(f"Semantic search disabled after error: {exc}")
            self._semantic_disabled = True
            return []

    def _bm25_search(self, query: str, allowed_ids: set[str] | None) -> list[dict]:
        if self._bm25 is None:
            return []

        q_low = query.lower()
        tokens = re.sub(r"[^a-z0-9]", " ", q_low).split()
        if any(term in q_low for term in ("strength", "load", "force", "withstand")):
            tokens.extend(["test", "load", "force", "dan", "traction", "tractive"])
        all_scores = self._bm25.get_scores(tokens)

        ranked = []
        for i, score in enumerate(all_scores):
            chunk = self._bm25_chunks[i]
            cid = chunk.get("chunk_id", "")
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            if score > 0:
                text = (
                    f"{chunk.get('heading_path', '')} "
                    f"{chunk.get('section_title', '')} "
                    f"{chunk.get('text', '')}"
                ).lower()
                boost = 1.0

                if any(term in q_low for term in ("strength", "load", "force", "withstand")):
                    if "test load" in text:
                        boost += 2.5
                    if "tractive force" in text:
                        boost += 2.0
                    if "dan" in text:
                        boost += 1.5
                    if "6.4." in text:
                        boost += 1.0
                    if "test in configuration" in text:
                        boost += 0.8

                    # De-prioritize contents / admin sections for requirements questions.
                    if "contents page" in text or "application for approval" in text:
                        boost *= 0.35
                    if "production definitively discontinued" in text:
                        boost *= 0.25
                    if "as from" in text and "contracting parties" in text:
                        boost *= 0.25

                ranked.append((float(score) * boost, i))

        ranked.sort(key=lambda x: -x[0])
        ranked = ranked[:TOP_K_CHUNKS]

        return [
            {
                "id": self._bm25_chunks[i].get("chunk_id", ""),
                "score": float(s),
                "bm25_score": float(s),
                "text": self._bm25_chunks[i].get("text", ""),
                "title": self._bm25_chunks[i].get("section_title", ""),
                "heading_path": self._bm25_chunks[i].get("heading_path", ""),
                "regulation": self._bm25_chunks[i].get("regulation", ""),
                "chunk_type": self._bm25_chunks[i].get("chunk_type", ""),
                "parent_id": self._bm25_chunks[i].get("parent_id"),
                "source": "bm25",
            }
            for s, i in ranked
        ]

    @staticmethod
    def _rrf_fusion(
        semantic: list[dict],
        bm25: list[dict],
        k: int = 60,
    ) -> list[dict]:
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, doc in enumerate(semantic):
            did = doc["id"]
            scores[did] = scores.get(did, 0) + SEMANTIC_WEIGHT / (k + rank + 1)
            docs[did] = doc

        for rank, doc in enumerate(bm25):
            did = doc["id"]
            scores[did] = scores.get(did, 0) + BM25_WEIGHT / (k + rank + 1)
            docs.setdefault(did, doc)

        merged = []
        for did, rrf_score in sorted(scores.items(), key=lambda x: -x[1]):
            d = docs[did].copy()
            d["rrf_score"] = rrf_score
            d["score"] = rrf_score
            merged.append(d)
        return merged[:TOP_K_RETRIEVE]

    @staticmethod
    def _multi_query_fusion(result_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
        """Fuse ranked lists from several query variants with RRF."""
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}
        for ranked in result_lists:
            for rank, doc in enumerate(ranked):
                did = doc["id"]
                scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
                docs.setdefault(did, doc)
        merged = []
        for did, sc in sorted(scores.items(), key=lambda x: -x[1]):
            d = docs[did].copy()
            d["mq_score"] = sc
            d["score"] = sc
            merged.append(d)
        return merged

    def _apply_metadata_boost(self, docs: list[dict], intent_flags: list[str]) -> list[dict]:
        """
        Metadata filtering stage: boost chunks whose metadata flags match the
        query intent and de-prioritize admin / contents / section-preview chunks.
        Soft boost (not a hard filter) so we never zero-out the candidate set.
        """
        if not ENABLE_METADATA_FILTER:
            return docs
        for d in docs:
            chunk = self._chunk_by_id.get(d["id"], {})
            mult = 1.0
            for flag in intent_flags:
                if chunk.get(flag):
                    mult += METADATA_BOOST
            # Prefer concrete paragraph content over section previews.
            if chunk.get("chunk_type") == "section":
                mult *= 0.85
            text_low = (chunk.get("text", "") or "").lower()
            if "contents page" in text_low or "application for approval" in text_low:
                mult *= 0.5
            d["score"] = d.get("score", 0.0) * mult
            d["metadata_mult"] = round(mult, 3)
        docs.sort(key=lambda x: -x.get("score", 0.0))
        return docs

    def _expand_parent_child(self, docs: list[dict]) -> list[dict]:
        """
        Parent-Child retrieval: precise child (paragraph) chunks are matched, then
        enriched with their parent section context. Near-duplicate section/paragraph
        variants of the same content are de-duplicated.
        """
        if not ENABLE_PARENT_CHILD:
            return docs

        seen_hashes: set[str] = set()
        deduped: list[dict] = []
        for d in docs:
            chunk = self._chunk_by_id.get(d["id"], {})
            # Normalize body (drop "# heading" / "[heading]" markers) so a section
            # preview and its child paragraph of the same clause collapse to one.
            body = re.sub(r"^[#\[].*?\]?\n+", "", (d.get("text", "") or ""), count=1)
            body_key = re.sub(r"[^a-z0-9]", "", body.lower())[:160]
            h = body_key or chunk.get("chunk_hash") or d.get("id", "")
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            parent_id = chunk.get("parent_id")
            if parent_id and parent_id in self._chunk_by_id:
                parent = self._chunk_by_id[parent_id]
                d["parent_id"] = parent_id
                d["parent_heading"] = parent.get("heading_path", "")
                # Attach concise parent context for the reranker/LLM.
                parent_text = (parent.get("text", "") or "").strip()
                if parent_text:
                    d["parent_context"] = parent_text[:600]
            deduped.append(d)
        return deduped

    def retrieve(self, query: str) -> dict[str, Any]:
        t0 = time.perf_counter()

        # 1. Query Expansion
        exp = expand_query(query) if ENABLE_QUERY_EXPANSION else None
        intent_flags = exp.intent_flags if exp else []

        # 2. Multi-Query Generation
        if ENABLE_MULTI_QUERY:
            queries = generate_multi_queries(query, exp)
        elif exp is not None:
            queries = [exp.expanded]
        else:
            queries = [query]

        # Metadata filter (regulation scope) applied to all variants.
        regs = self._detect_regs(query)
        allowed = self._filter_chunk_ids(regs)

        # 3. Hybrid Retrieval (Dense + BM25 + RRF) per query variant
        per_query: list[list[dict]] = []
        semantic_total = 0
        bm25_total = 0
        for q in queries:
            semantic = self._semantic_search(q, allowed)
            bm25 = self._bm25_search(q, allowed)
            semantic_total += len(semantic)
            bm25_total += len(bm25)
            per_query.append(self._rrf_fusion(semantic, bm25, k=RRF_K))

        fused = (
            self._multi_query_fusion(per_query, k=RRF_K)
            if len(per_query) > 1
            else per_query[0]
        )

        # 4. Metadata Filtering / boosting by intent
        fused = self._apply_metadata_boost(fused, intent_flags)
        fused = fused[:TOP_K_RETRIEVE]

        # 5. Parent-Child Retrieval (dedupe + attach parent context)
        fused = self._expand_parent_child(fused)

        # Best raw semantic similarity in the candidate set (grounding signal).
        sem_scores = [
            d["semantic_score"] for d in fused if d.get("semantic_score") is not None
        ]
        top_semantic_score = max(sem_scores) if sem_scores else None

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"Retrieve done: queries={len(queries)} semantic={semantic_total} "
            f"bm25={bm25_total} fused={len(fused)} intent={intent_flags} in {latency_ms}ms"
        )
        return {
            "query": query,
            "queries": queries,
            "documents": fused,
            "semantic_count": semantic_total,
            "bm25_count": bm25_total,
            "intent_flags": intent_flags,
            "top_semantic_score": top_semantic_score,
            "latency_ms": latency_ms,
        }
