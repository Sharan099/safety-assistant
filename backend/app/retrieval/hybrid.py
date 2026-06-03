"""
Hybrid retriever: semantic (MiniLM) + BM25 with Reciprocal Rank Fusion.
Vectorized cosine search over precomputed embeddings.
"""

import json
import os
import re
import time
from typing import Any

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from backend.app.core.settings import (
    BM25_WEIGHT,
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    EMBEDDING_MODEL,
    RRF_K,
    SEMANTIC_WEIGHT,
    TOP_K_CHUNKS,
    TOP_K_RETRIEVE,
    TOP_K_VECTOR,
    VECTOR_SCORE_THRESHOLD,
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


class HybridRetriever:
    def __init__(self) -> None:
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
        if CHUNKS_FILE.exists():
            with open(CHUNKS_FILE, encoding="utf-8") as f:
                self.chunks = json.load(f).get("chunks", [])
            self._chunk_by_id = {
                c.get("chunk_id", ""): c for c in self.chunks if c.get("chunk_id")
            }
            logger.info(f"Loaded {len(self.chunks)} chunks")

        if EMBEDDINGS_FILE.exists():
            with open(EMBEDDINGS_FILE, encoding="utf-8") as f:
                data = json.load(f)
                self.embeddings = data.get("embeddings", {})
            logger.info(f"Loaded {len(self.embeddings)} embeddings")

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
                self._model = SentenceTransformer(EMBEDDING_MODEL)
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
            q = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
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
                        "text": c.get("text", ""),
                        "title": c.get("section_title", ""),
                        "heading_path": c.get("heading_path", ""),
                        "regulation": c.get("regulation", ""),
                        "chunk_type": c.get("chunk_type", ""),
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
                "text": self._bm25_chunks[i].get("text", ""),
                "title": self._bm25_chunks[i].get("section_title", ""),
                "heading_path": self._bm25_chunks[i].get("heading_path", ""),
                "regulation": self._bm25_chunks[i].get("regulation", ""),
                "chunk_type": self._bm25_chunks[i].get("chunk_type", ""),
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

    def retrieve(self, query: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        regs = self._detect_regs(query)
        allowed = self._filter_chunk_ids(regs)

        semantic = self._semantic_search(query, allowed)
        bm25 = self._bm25_search(query, allowed)
        fused = self._rrf_fusion(semantic, bm25, k=RRF_K)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"Retrieve done: semantic={len(semantic)} bm25={len(bm25)} "
            f"fused={len(fused)} in {latency_ms}ms"
        )
        return {
            "query": query,
            "documents": fused,
            "semantic_count": len(semantic),
            "bm25_count": len(bm25),
            "latency_ms": latency_ms,
        }
