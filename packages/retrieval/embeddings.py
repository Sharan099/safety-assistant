"""EmbeddingProvider — TRD.md §20, PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md §5/§16.

Three tiers, matching the final-fix doc's "local / remote / mock" provider
architecture — business logic (`packages/retrieval/search.py`) depends only
on the `EmbeddingProvider` Protocol, never on a specific tier:

- **local** (`FastEmbedProvider`, production default as of `docs/ADR/0014`):
  real semantic embeddings via `fastembed` — ONNX Runtime, not
  `sentence-transformers`/`torch`. `docs/ADR/0011` deferred Docling/a real
  reranker specifically because `torch` risks disk exhaustion (12 GB free
  measured); `fastembed`'s small quantized models (~70-90 MB) carry none of
  that risk and were benchmarked, not assumed, to beat the hashing
  placeholder — see `evals/embedding_benchmark.py` /
  `evals/results/embedding_benchmark.json`.
- **remote**: no concrete implementation. `LLM_BASE_URL`/`LLM_API_KEY`
  (`docs/ADR/0003`) point at FreeLLMAPI, but no credentials are configured
  in this environment (`.env` doesn't exist here) — building a remote
  provider with no way to test it against a real endpoint would risk
  exactly the kind of silent, unverified bug the "never silently substitute
  a fake capability" rule exists to prevent. The `EmbeddingProvider`
  Protocol is the only integration point a real remote implementation would
  need; nothing above this module changes when one is added.
- **mock** (`HashingEmbeddingProvider`, formerly the interim default):
  deterministic, dependency-free, fully offline feature-hashed
  bag-of-words. Kept for tests — no network, no model download, no
  variance.
"""

from __future__ import annotations

import functools
import hashlib
import math
import re
from typing import Protocol

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(Protocol):
    model_name: str
    model_version: str
    dimensions: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbeddingProvider:
    """The "mock" tier — see module docstring."""

    model_name = "hashing-bow"
    model_version = "v1"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


# Chosen by evals/embedding_benchmark.py (docs/ADR/0014), not by leaderboard
# reputation: measured MRR 0.838 vs BAAI/bge-small-en-v1.5's 0.729 and the
# hashing placeholder's 0.249 on the real golden set, with faster indexing
# too (22.9s vs 143.8s for the same 475 real chunks). Both real candidates
# tied on Recall@5/@10 (1.00) — MRR was the deciding metric.
DEFAULT_FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class FastEmbedProvider:
    """The "local" real-semantic tier — see module docstring. Loads the
    ONNX model once per instance (~1-2s); `embed()` itself is fast
    (tens of ms) since the model is already resident."""

    model_version = "v1"

    def __init__(self, model_name: str = DEFAULT_FASTEMBED_MODEL) -> None:
        # Local import: avoids paying ONNX Runtime's import cost for callers
        # that only ever construct HashingEmbeddingProvider.
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dimensions = next(m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == model_name)

    def embed(self, text: str) -> list[float]:
        (vector,) = self._model.embed([text])
        return vector.tolist()  # type: ignore[no-any-return]


@functools.lru_cache(maxsize=1)
def get_default_embedding_provider() -> FastEmbedProvider:
    """A process-wide singleton — loading the ONNX model costs ~1-2s;
    `packages/retrieval/search.py`'s `retrieve()`/`vector_search()` are
    called once per API request/Copilot turn, so reloading it fresh every
    call would be pure waste for identical output."""
    return FastEmbedProvider()
