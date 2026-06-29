import os
import re
import torch
from typing import List
from loguru import logger

from registry.embedding_config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_DOC_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_TRUST_REMOTE_CODE,
    verify_embedding_dimension,
)


def _mean_pool(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class RegulationEmbedder:
    """
    Dense embeddings pinned to nomic-ai/nomic-embed-text-v1.5 (768-d).
    Uses HuggingFace transformers on Windows (SentenceTransformer can segfault);
    SentenceTransformer on other platforms when available.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.dimension = EMBEDDING_DIMENSION
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backend = "mock"
        self.model = None
        self._hf_model = None
        self._hf_tokenizer = None

        env_mock = os.getenv("USE_MOCK_EMBEDDINGS", "").lower()
        if env_mock == "false":
            self.use_mock_embeddings = False
        elif env_mock == "true" or (os.name == "nt" and env_mock != "false"):
            logger.warning(
                "Windows detected or USE_MOCK_EMBEDDINGS=true. "
                "Using deterministic mock embeddings at dim=%s.",
                self.dimension,
            )
            self.use_mock_embeddings = True
            return
        else:
            self.use_mock_embeddings = False

        if os.name == "nt" or os.getenv("EMBED_BACKEND", "").lower() == "transformers":
            self._init_transformers_backend()
        else:
            self._init_sentence_transformer_backend()

    def _init_transformers_backend(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(
                "Initializing RegulationEmbedder (transformers) %s on %s",
                self.model_name,
                self.device,
            )
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE
            )
            self._hf_model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE
            )
            self._hf_model.to(self.device)
            self._hf_model.eval()
            verify_embedding_dimension(self.dimension)
            self.backend = "transformers"
        except Exception as exc:
            logger.warning("Transformers embedder failed: %s. Falling back to mock.", exc)
            self.use_mock_embeddings = True
            self.backend = "mock"

    def _init_sentence_transformer_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Initializing RegulationEmbedder (sentence-transformers) %s on %s",
                self.model_name,
                self.device,
            )
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE,
            )
            verify_embedding_dimension(self.model.get_sentence_embedding_dimension())
            self.backend = "sentence_transformers"
        except Exception as exc:
            logger.warning("SentenceTransformer failed (%s); trying transformers.", exc)
            self._init_transformers_backend()

    def _prefix_passages(self, texts: List[str]) -> List[str]:
        if "nomic" in self.model_name.lower():
            return [f"{EMBEDDING_DOC_PREFIX}{t}" for t in texts]
        if "bge" in self.model_name.lower():
            return texts
        return texts

    def _prefix_query(self, query: str) -> str:
        if "nomic" in self.model_name.lower():
            return f"{EMBEDDING_QUERY_PREFIX}{query}"
        if "bge" in self.model_name.lower():
            return f"Represent this sentence for searching relevant passages: {query}"
        return query

    def _encode_transformers(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        assert self._hf_model is not None and self._hf_tokenizer is not None
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._hf_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._hf_model(**inputs)
            pooled = _mean_pool(outputs, inputs["attention_mask"])
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.extend(normed.cpu().numpy().tolist())
        return out

    def embed_chunks(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []

        if self.use_mock_embeddings:
            return [self._generate_mock_embedding(t) for t in texts]

        passages = self._prefix_passages(texts)
        try:
            if self.backend == "transformers":
                return self._encode_transformers(passages, batch_size=min(batch_size, 8))
            embeddings = self.model.encode(
                passages,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}. Falling back to mock embeddings.")
            return [self._generate_mock_embedding(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        if self.use_mock_embeddings:
            return self._generate_mock_embedding(query)

        query_text = self._prefix_query(query)
        try:
            if self.backend == "transformers":
                return self._encode_transformers([query_text], batch_size=1)[0]
            embedding = self.model.encode(
                query_text,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding query failed: {e}. Falling back to mock embedding.")
            return self._generate_mock_embedding(query)

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Deterministic unit-length mock vector matching pinned dimension."""
        import hashlib
        import numpy as np

        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h, "big") % (2**32 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0.0, 1.0, self.dimension)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class RegulationReranker:
    """Cross-encoder reranker (BAAI/bge-reranker-v2-m3) with keyword fallback."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if os.name == "nt" or os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true":
            logger.warning(
                "Windows detected or USE_MOCK_EMBEDDINGS=true. Using keyword overlap reranking."
            )
            self.use_mock_reranker = True
            self.model = None
            return

        self.use_mock_reranker = False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Initializing RegulationReranker with {self.model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Could not load Reranker model {self.model_name}: {e}. Using keyword fallback.")
            self.use_mock_reranker = True
            self.model = None

    def compute_scores(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []

        if self.use_mock_reranker or not self.model:
            scores = []
            query_words = set(re.sub(r"[^\w\s]", "", query).lower().split())
            for p in passages:
                p_words = set(p.lower().split())
                scores.append(float(len(query_words.intersection(p_words))))
            return scores

        pairs = [[query, p] for p in passages]
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                return outputs.logits.view(-1).float().cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to keyword scoring.")
            query_words = set(query.lower().split())
            return [
                float(len(query_words.intersection(set(p.lower().split()))))
                for p in passages
            ]
