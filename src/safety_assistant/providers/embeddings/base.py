"""Typed embedding interface. Business logic depends on this Protocol only."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    model_version: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
