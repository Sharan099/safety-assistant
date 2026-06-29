"""Ragas embedding adapter reusing the pinned Nomic RegulationEmbedder."""

from __future__ import annotations

from ragas.embeddings.base import BaseRagasEmbeddings

from vectorization.embedder import RegulationEmbedder


class RegistryNomicEmbeddings(BaseRagasEmbeddings):
    """Reuse live corpus embedder — avoids loading a second HF model on Windows."""

    def __init__(self, embedder: RegulationEmbedder | None = None) -> None:
        self._embedder = embedder if embedder is not None else RegulationEmbedder()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embedder.embed_query(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)
