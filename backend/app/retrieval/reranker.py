"""Cross-encoder and Jina-v3 reranking over hybrid retrieval candidates."""

import os
import time
from typing import Any

from loguru import logger

from backend.app.core.settings import ENABLE_RERANKER, RERANKER_MODEL, TOP_K_AFTER_RERANK

# auto | crossencoder | jina | qwen — "auto" picks from RERANKER_MODEL name
RERANKER_KIND = os.getenv("RERANKER_KIND", "auto").lower()


def _doc_text(doc: dict) -> str:
    return (
        f"{doc.get('heading_path') or doc.get('title', '')} "
        f"{doc.get('parent_context', '')[:200]} "
        f"{doc.get('text', '')[:800]}"
    ).strip()


def _resolve_kind() -> str:
    if RERANKER_KIND != "auto":
        return RERANKER_KIND
    name = RERANKER_MODEL.lower()
    if "jina" in name:
        return "jina"
    if "qwen" in name:
        return "qwen"
    return "crossencoder"


class CrossEncoderReranker:
    """Production reranker: BGE/Qwen (CrossEncoder) or Jina-v3 (transformers)."""

    def __init__(self) -> None:
        self._model = None
        self._available = True
        self._enabled = ENABLE_RERANKER
        self._kind = _resolve_kind()

    def _load(self) -> None:
        if self._model is not None or not self._enabled:
            return
        try:
            if self._kind == "jina":
                from transformers import AutoModel

                self._model = AutoModel.from_pretrained(
                    RERANKER_MODEL,
                    trust_remote_code=True,
                )
                self._model.eval()
            else:
                from sentence_transformers import CrossEncoder

                trust = self._kind == "qwen"
                self._model = CrossEncoder(
                    RERANKER_MODEL,
                    trust_remote_code=trust,
                )
                if trust and getattr(self._model, "tokenizer", None) is not None:
                    tok = self._model.tokenizer
                    if tok.pad_token is None and tok.eos_token:
                        tok.pad_token = tok.eos_token
                    if hasattr(self._model, "model") and hasattr(
                        self._model.model, "config"
                    ):
                        self._model.model.config.pad_token_id = tok.pad_token_id
            logger.info(f"Reranker loaded: {RERANKER_MODEL} (kind={self._kind})")
        except Exception as exc:
            logger.warning(f"Reranker unavailable ({RERANKER_MODEL}): {exc}")
            self._available = False

    def warmup(self) -> None:
        if not self._enabled:
            return
        self._load()
        if self._model is None:
            return
        if self._kind == "jina":
            self._model.rerank(
                query="warmup",
                documents=["warmup document"],
                top_n=1,
            )
        else:
            self._model.predict([("warmup", "warmup document")])
        logger.info("Reranker warmed up")

    def _fallback(self, documents: list[dict], t0: float) -> dict[str, Any]:
        top = documents[:TOP_K_AFTER_RERANK]
        for d in top:
            d["rerank_score"] = d.get("rrf_score", d.get("score", 0))
        return {
            "documents": top,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "reranker_used": False,
        }

    def rerank(self, query: str, documents: list[dict]) -> dict[str, Any]:
        t0 = time.perf_counter()
        if not documents:
            return {"documents": [], "latency_ms": 0, "reranker_used": False}

        if not self._enabled:
            return self._fallback(documents, t0)

        self._load()
        if not self._available or self._model is None:
            return self._fallback(documents, t0)

        docs_copy = [dict(d) for d in documents]
        texts = [_doc_text(d) for d in docs_copy]

        try:
            if self._kind == "jina":
                ranked = self._model.rerank(
                    query=query,
                    documents=texts,
                    top_n=TOP_K_AFTER_RERANK,
                )
                for r in ranked:
                    idx = int(r.get("index", r.get("corpus_id", 0)))
                    score = float(
                        r.get("relevance_score", r.get("score", r.get("relevance", 0)))
                    )
                    if 0 <= idx < len(docs_copy):
                        docs_copy[idx]["rerank_score"] = score
                ranked_docs = sorted(
                    docs_copy,
                    key=lambda x: x.get("rerank_score", 0),
                    reverse=True,
                )[:TOP_K_AFTER_RERANK]
            else:
                pairs = [(query, t) for t in texts]
                scores = self._model.predict(pairs, show_progress_bar=False)
                for doc, score in zip(docs_copy, scores):
                    doc["rerank_score"] = float(score)
                ranked_docs = sorted(
                    docs_copy,
                    key=lambda x: x.get("rerank_score", 0),
                    reverse=True,
                )[:TOP_K_AFTER_RERANK]
        except Exception as exc:
            logger.warning(f"Rerank predict failed: {exc}")
            return self._fallback(documents, t0)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"Rerank done: top={len(ranked_docs)} in {latency_ms}ms")
        return {
            "documents": ranked_docs,
            "latency_ms": latency_ms,
            "reranker_used": True,
        }
