"""Incremental embed/index — reuse unchanged chunk embeddings by content_hash (FR-23)."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from database.models import Chunk
from registry.embedding_config import EMBEDDING_DIMENSION, verify_embedding_dimension
from vectorization.embedder import RegulationEmbedder


def load_reusable_embeddings(db: Session, document_id: int) -> dict[str, list[float]]:
    """Map content_hash → embedding for chunks already indexed on this document."""
    rows = (
        db.query(Chunk.content_hash, Chunk.embedding)
        .filter(Chunk.document_id == document_id, Chunk.content_hash.isnot(None))
        .all()
    )
    reusable: dict[str, list[float]] = {}
    for content_hash, embedding in rows:
        if not content_hash or not embedding:
            continue
        vec = embedding if isinstance(embedding, list) else json.loads(embedding)
        if len(vec) == EMBEDDING_DIMENSION:
            reusable[content_hash] = vec
    return reusable


def embed_chunks_incremental(
    embedder: RegulationEmbedder,
    chunks: list[dict[str, Any]],
    reusable: dict[str, list[float]],
) -> tuple[list[list[float]], dict[str, int]]:
    """
    Attach embeddings to chunks, reusing vectors when content_hash is unchanged.

    Returns (embeddings aligned to chunks, stats).
    """
    embeddings: list[list[float] | None] = [None] * len(chunks)
    to_embed_texts: list[str] = []
    to_embed_indices: list[int] = []
    reused = 0

    for idx, chunk in enumerate(chunks):
        content_hash = chunk.get("content_hash")
        if content_hash and content_hash in reusable:
            embeddings[idx] = reusable[content_hash]
            reused += 1
            continue
        to_embed_indices.append(idx)
        to_embed_texts.append(chunk["chunk_text"])

    embedded = 0
    if to_embed_texts:
        new_vectors = embedder.embed_chunks(to_embed_texts)
        embedded = len(new_vectors)
        for idx, vec in zip(to_embed_indices, new_vectors):
            verify_embedding_dimension(len(vec))
            embeddings[idx] = vec

    if any(e is None for e in embeddings):
        raise RuntimeError("Incremental embed left gaps — all chunks must have embeddings")

    stats = {
        "chunks_total": len(chunks),
        "embedded_new": embedded,
        "reused": reused,
    }
    logger.info(
        "Incremental embed: %s new, %s reused of %s chunks",
        embedded,
        reused,
        len(chunks),
    )
    return embeddings, stats  # type: ignore[return-value]


def prepare_chunks_for_index(
    db: Session,
    document_id: int,
    chunks: list[dict[str, Any]],
    embedder: RegulationEmbedder,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Embed chunks incrementally and attach vectors in-place."""
    if not chunks:
        return [], {"chunks_total": 0, "embedded_new": 0, "reused": 0}

    reusable = load_reusable_embeddings(db, document_id)
    vectors, stats = embed_chunks_incremental(embedder, chunks, reusable)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    return chunks, stats
