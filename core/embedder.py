"""CPU-aware dense embeddings and cross-encoder reranker."""

from __future__ import annotations

import os
import re

import torch
from loguru import logger

from app.config import settings

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", settings.EMBEDDING_MODEL)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", str(settings.EMBEDDING_DIMENSION)))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", settings.RERANKER_MODEL)
RERANKER_MAX_PASSAGE_CHARS = int(
    os.getenv("RERANKER_MAX_PASSAGE_CHARS", str(settings.RERANKER_MAX_PASSAGE_CHARS))
)
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "32"))

_reranker_singleton: "Reranker | None" = None


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.dimension = EMBEDDING_DIMENSION
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._init()

    def _init(self) -> None:
        from sentence_transformers import SentenceTransformer

        offline = os.getenv("HF_HUB_OFFLINE") == "1"
        logger.info(
            "Loading embedder {} on {} (HF_HUB_OFFLINE={})", self.model_name, self.device, offline
        )
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001 - re-raise with actionable context
            hint = (
                "Model is not present in the local Hugging Face cache and "
                "HF_HUB_OFFLINE=1 forbids downloading it. Rebuild the image "
                "(the Dockerfile prefetches this model at build time) or "
                "temporarily set HF_HUB_OFFLINE=0 to allow a one-off download."
                if offline
                else "Could not download the model from Hugging Face Hub — "
                "check network egress to huggingface.co, or pre-bake the "
                "model into the image (see Dockerfile) to avoid depending on "
                "runtime network access altogether."
            )
            logger.error("Failed to load embedder {}: {}. {}", self.model_name, exc, hint)
            raise RuntimeError(f"Embedder load failed for {self.model_name}: {exc}. {hint}") from exc
        dim = self._model.get_sentence_embedding_dimension()
        if dim != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: model {self.model_name} outputs {dim}d "
                f"but EMBEDDING_DIMENSION={self.dimension}. "
                "Align .env / docker-compose with the database schema."
            )
        logger.info("Embedder {} loaded successfully ({}d)", self.model_name, dim)

    def _doc_text(self, text: str) -> str:
        if "bge" in self.model_name.lower():
            return text
        return text

    def _query_text(self, query: str) -> str:
        if "bge" in self.model_name.lower():
            return f"Represent this sentence for searching relevant passages: {query}"
        return query

    def embed_passages(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        if not texts:
            return []
        passages = [self._doc_text(t) for t in texts]
        vectors = self._model.encode(
            passages,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self._model.encode(
            self._query_text(query),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.tolist()


class Reranker:
    """Cross-encoder reranker (model name from RERANKER_MODEL env var)."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or RERANKER_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        offline = os.getenv("HF_HUB_OFFLINE") == "1"
        logger.info(
            "Loading reranker {} on {} (HF_HUB_OFFLINE={})", self.model_name, self.device, offline
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except Exception as exc:  # noqa: BLE001 - re-raise with actionable context
            hint = (
                "Model is not present in the local Hugging Face cache and "
                "HF_HUB_OFFLINE=1 forbids downloading it. Rebuild the image "
                "(the Dockerfile prefetches this model at build time) or "
                "temporarily set HF_HUB_OFFLINE=0 to allow a one-off download."
                if offline
                else "Could not download the model from Hugging Face Hub — "
                "check network egress to huggingface.co, or pre-bake the "
                "model into the image (see Dockerfile) to avoid depending on "
                "runtime network access altogether."
            )
            logger.error("Failed to load reranker {}: {}. {}", self.model_name, exc, hint)
            raise RuntimeError(f"Reranker load failed for {self.model_name}: {exc}. {hint}") from exc
        self.model.to(self.device)
        self.model.eval()
        logger.info("Reranker {} loaded successfully", self.model_name)

    @staticmethod
    def prepare_passage(text: str, max_chars: int | None = None) -> str:
        """Trim passage text before cross-encoder tokenization (rerank-only, not retrieval)."""
        limit = max_chars or RERANKER_MAX_PASSAGE_CHARS
        text = re.sub(r"\s+", " ", (text or "").strip())
        if len(text) <= limit:
            return text
        half = limit // 2
        return text[:half] + " … " + text[-half:]

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        trimmed = [self.prepare_passage(p) for p in passages]
        pairs = [[query, p] for p in trimmed]
        all_scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), RERANKER_BATCH_SIZE):
                batch = pairs[start : start + RERANKER_BATCH_SIZE]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits.view(-1)
                all_scores.extend(logits.float().cpu().tolist())
        return all_scores


def get_reranker() -> Reranker:
    """Process-wide singleton — loaded once at startup, not per request."""
    global _reranker_singleton
    if _reranker_singleton is None:
        _reranker_singleton = Reranker()
    return _reranker_singleton


def reset_reranker_for_tests() -> None:
    global _reranker_singleton
    _reranker_singleton = None
