"""EmbeddingProvider — TRD.md §20.

"Do not commit to an embedding model before benchmarking." `HashingEmbeddingProvider`
is the interim default: deterministic, dependency-free, fully offline
feature-hashed bag-of-words. It exists so `packages/retrieval` works
end-to-end (real pgvector storage, real cosine search, real RRF fusion)
without pulling in a multi-hundred-MB model (sentence-transformers/torch) on
an 8 GB RAM dev machine before that benchmark happens — see docs/ADR/0007.

Retrieval quality is modest (word-overlap only, no semantics). Swap in a
real model by implementing this same `Protocol` once TRD.md §20's benchmark
picks one; nothing above this layer needs to change.
"""

from __future__ import annotations

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
