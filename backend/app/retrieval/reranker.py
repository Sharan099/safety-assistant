"""Cross-encoder reranking over hybrid retrieval candidates."""

import os
import time
from typing import Any

from loguru import logger

from backend.app.core.settings import RERANKER_MODEL, TOP_K_AFTER_RERANK


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model = None
        self._available = True
        self._enabled = os.getenv("ENABLE_RERANKER", "false").lower() == "true"

    def _load(self) -> None:
        if self._model is None and self._enabled:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(RERANKER_MODEL)
                logger.info(f"Reranker loaded: {RERANKER_MODEL}")
            except Exception as exc:
                logger.warning(f"Reranker unavailable: {exc}")
                self._available = False

    def warmup(self) -> None:
        if not self._enabled:
            return
        self._load()
        if self._model is not None:
            self._model.predict([("warmup", "warmup document")])
            logger.info("Reranker warmed up")

    def rerank(self, query: str, documents: list[dict]) -> dict[str, Any]:
        t0 = time.perf_counter()
        if not documents:
            return {"documents": [], "latency_ms": 0, "reranker_used": False}

        if not self._enabled:
            top = documents[:TOP_K_AFTER_RERANK]
            for d in top:
                d["rerank_score"] = d.get("rrf_score", d.get("score", 0))
            return {
                "documents": top,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "reranker_used": False,
            }

        self._load()
        if not self._available or self._model is None:
            top = documents[:TOP_K_AFTER_RERANK]
            for d in top:
                d["rerank_score"] = d.get("rrf_score", d.get("score", 0))
            return {
                "documents": top,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "reranker_used": False,
            }

        pairs = [
            (
                query,
                f"{d.get('heading_path') or d.get('title', '')} "
                f"{d.get('parent_context', '')[:200]} "
                f"{d.get('text', '')[:800]}".strip(),
            )
            for d in documents
        ]
        scores = self._model.predict(pairs, show_progress_bar=False)
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        ranked = sorted(
            documents,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True,
        )[:TOP_K_AFTER_RERANK]

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"Rerank done: top={len(ranked)} in {latency_ms}ms")
        return {
            "documents": ranked,
            "latency_ms": latency_ms,
            "reranker_used": True,
        }
