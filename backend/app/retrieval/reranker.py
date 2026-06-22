"""Cross-encoder and Jina-v3 reranking over hybrid retrieval candidates."""

import os
import time
from typing import Any

from loguru import logger

from backend.app.core.settings import (
    COMPARISON_CHUNKS_PER_REG,
    ENABLE_RERANKER,
    RERANKER_MODEL,
    TOP_K_AFTER_RERANK,
)

# auto | crossencoder | jina | qwen — "auto" picks from RERANKER_MODEL name
RERANKER_KIND = os.getenv("RERANKER_KIND", "auto").lower()
RERANKER_REVISION = os.getenv("RERANKER_REVISION", "").strip() or None

_VALID_KINDS = frozenset({"auto", "crossencoder", "jina", "qwen"})


def _doc_text(doc: dict) -> str:
    return (
        f"{doc.get('heading_path') or doc.get('title', '')} "
        f"{doc.get('parent_context', '')[:200]} "
        f"{doc.get('text', '')[:800]}"
    ).strip()


def _resolve_kind() -> str:
    kind = os.getenv("RERANKER_KIND", RERANKER_KIND).lower()
    model_name = os.getenv("RERANKER_MODEL", RERANKER_MODEL).lower()
    if kind not in _VALID_KINDS:
        # Common HF misconfig: model id pasted into RERANKER_KIND instead of RERANKER_MODEL
        logger.warning(
            f"RERANKER_KIND={kind!r} is invalid (use auto|crossencoder|jina|qwen); "
            f"inferring loader from RERANKER_MODEL={model_name!r}"
        )
        kind = "auto"
    if kind != "auto":
        return kind
    if "jina" in model_name:
        return "jina"
    if "qwen" in model_name:
        return "qwen"
    return "crossencoder"


class CrossEncoderReranker:
    """Production reranker: BGE/Qwen (CrossEncoder) or Jina-v3 (transformers)."""

    def __init__(self) -> None:
        self._model = None
        self._available = True
        self._enabled = ENABLE_RERANKER
        self._kind: str | None = None

    @property
    def kind(self) -> str:
        if self._kind is None:
            self._kind = _resolve_kind()
        return self._kind

    def _load(self) -> None:
        if self._model is not None or not self._enabled:
            return
        try:
            model_id = os.getenv("RERANKER_MODEL", RERANKER_MODEL)
            if self.kind == "jina":
                from transformers import AutoModel

                self._model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    revision=RERANKER_REVISION,
                )
                self._model.eval()
            else:
                from sentence_transformers import CrossEncoder

                trust = self.kind == "qwen"
                kwargs: dict[str, Any] = {"trust_remote_code": trust}
                if RERANKER_REVISION:
                    kwargs["revision"] = RERANKER_REVISION
                self._model = CrossEncoder(model_id, **kwargs)
                if trust and getattr(self._model, "tokenizer", None) is not None:
                    tok = self._model.tokenizer
                    if tok.pad_token is None and tok.eos_token:
                        tok.pad_token = tok.eos_token
                    if hasattr(self._model, "model") and hasattr(
                        self._model.model, "config"
                    ):
                        self._model.model.config.pad_token_id = tok.pad_token_id
            logger.info(f"Reranker loaded: {model_id} (kind={self.kind})")
        except Exception as exc:
            logger.warning(f"Reranker unavailable ({RERANKER_MODEL}): {exc}")
            self._available = False

    def warmup(self) -> None:
        if not self._enabled:
            return
        self._load()
        if self._model is None:
            return
        if self.kind == "jina":
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

    def _balanced_comparison_select(self, docs_copy: list[dict]) -> list[dict]:
        """Keep top chunks per regulation when comparison_reg is tagged."""
        by_reg: dict[str, list[dict]] = {}
        for d in docs_copy:
            reg = d.get("comparison_reg") or d.get("regulation") or "unknown"
            by_reg.setdefault(str(reg), []).append(d)
        if len(by_reg) < 2:
            return sorted(
                docs_copy, key=lambda x: x.get("rerank_score", 0), reverse=True
            )[:TOP_K_AFTER_RERANK]

        per_reg = max(1, TOP_K_AFTER_RERANK // len(by_reg))
        per_reg = min(per_reg, max(1, COMPARISON_CHUNKS_PER_REG))
        selected: list[dict] = []
        seen: set[str] = set()
        for items in by_reg.values():
            items.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            for d in items[:per_reg]:
                if d["id"] not in seen:
                    seen.add(d["id"])
                    selected.append(d)
        rest = sorted(
            [d for d in docs_copy if d["id"] not in seen],
            key=lambda x: x.get("rerank_score", 0),
            reverse=True,
        )
        for d in rest:
            if len(selected) >= TOP_K_AFTER_RERANK:
                break
            selected.append(d)
        return selected[:TOP_K_AFTER_RERANK]

    def _balanced_cluster_select(self, docs_copy: list[dict]) -> list[dict]:
        """Keep top chunks per cluster member when cluster_reg is tagged."""
        by_reg: dict[str, list[dict]] = {}
        for d in docs_copy:
            reg = d.get("cluster_reg") or d.get("regulation") or "unknown"
            by_reg.setdefault(str(reg), []).append(d)
        if len(by_reg) < 2:
            return sorted(
                docs_copy, key=lambda x: x.get("rerank_score", 0), reverse=True
            )[:TOP_K_AFTER_RERANK]

        per_reg = max(1, TOP_K_AFTER_RERANK // len(by_reg))
        selected: list[dict] = []
        seen: set[str] = set()
        for items in by_reg.values():
            items.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            for d in items[:per_reg]:
                if d["id"] not in seen:
                    seen.add(d["id"])
                    selected.append(d)
        rest = sorted(
            [d for d in docs_copy if d["id"] not in seen],
            key=lambda x: x.get("rerank_score", 0),
            reverse=True,
        )
        for d in rest:
            if len(selected) >= TOP_K_AFTER_RERANK:
                break
            selected.append(d)
        return selected[:TOP_K_AFTER_RERANK]

    def _finalize_ranked(self, ranked_docs: list[dict]) -> list[dict]:
        if any(d.get("comparison_reg") for d in ranked_docs):
            return self._balanced_comparison_select(ranked_docs)
        if any(d.get("cluster_reg") for d in ranked_docs):
            return self._balanced_cluster_select(ranked_docs)
        return ranked_docs[:TOP_K_AFTER_RERANK]

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
            if self.kind == "jina":
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
                )
            else:
                pairs = [(query, t) for t in texts]
                scores = self._model.predict(pairs, show_progress_bar=False)
                for doc, score in zip(docs_copy, scores):
                    doc["rerank_score"] = float(score)
                ranked_docs = sorted(
                    docs_copy,
                    key=lambda x: x.get("rerank_score", 0),
                    reverse=True,
                )
            ranked_docs = self._finalize_ranked(ranked_docs)
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
