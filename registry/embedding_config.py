"""Pinned embedding model settings — must match vectors stored in safety_registry.db."""

from __future__ import annotations

import os

# Live safety_registry.db (7650 chunks) was built with Nomic v1.5 @ 768-d (see docs/EMBEDDING_EVIDENCE.md).
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
EMBEDDING_QUERY_PREFIX: str = os.getenv("EMBEDDING_QUERY_PREFIX", "search_query: ")
EMBEDDING_DOC_PREFIX: str = os.getenv("EMBEDDING_DOC_PREFIX", "search_document: ")
EMBEDDING_TRUST_REMOTE_CODE: bool = (
    os.getenv("EMBEDDING_TRUST_REMOTE_CODE", "true").lower() == "true"
)


def verify_embedding_dimension(dimension: int) -> None:
    if dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: got {dimension}, "
            f"expected {EMBEDDING_DIMENSION} for model {EMBEDDING_MODEL}"
        )
