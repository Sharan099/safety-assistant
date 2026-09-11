"""Dense leg: pgvector cosine distance over the scoped candidate universe (HNSW index)."""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy.orm import Session

from safety_assistant.persistence.models import Chunk, ChunkEmbedding
from safety_assistant.providers.embeddings import EmbeddingProvider
from safety_assistant.retrieval.base import scoped_statement
from safety_assistant.retrieval.filters import ScopeFilter


def dense_search(
    session: Session,
    query_vector: list[float],
    scope: ScopeFilter,
    provider: EmbeddingProvider,
    *,
    top_k: int,
    today: datetime.date | None = None,
) -> list[tuple[uuid.UUID, float]]:
    """Ranked (chunk_id, cosine_distance) — lower distance is better."""
    distance = ChunkEmbedding.embedding.cosine_distance(query_vector)
    stmt = (
        scoped_statement(scope, today=today)
        .with_only_columns(Chunk.id, distance.label("distance"))
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .where(ChunkEmbedding.model_name == provider.model_name, ChunkEmbedding.model_version == provider.model_version)
        .order_by(distance.asc())
        .limit(top_k)
    )
    return [(cid, float(d)) for cid, d in session.execute(stmt).all()]
