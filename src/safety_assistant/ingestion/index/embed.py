"""Embedding/indexing stage with content-addressed reuse.

``embed  = chunk_sha256 + embedding_model_version + dimensions``

Before embedding a chunk we look for an existing embedding of *the same
content* (same sha256) under *any* version of the same regulation with the
same model — an amendment that leaves 95 % of clauses untouched re-embeds
5 %. Reuse is by value copy, so every chunk row still owns one embedding row
and cascades cleanly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from safety_assistant.persistence.models import Chunk, ChunkEmbedding, RegulationVersion
from safety_assistant.providers.embeddings import EmbeddingProvider

INDEX_SCHEMA_VERSION = 1
_BATCH = 64


@dataclass
class EmbedStats:
    total: int
    reused: int
    embedded: int


def snapshot_embeddings(
    session: Session, version: RegulationVersion, provider: EmbeddingProvider
) -> dict[str, list[float]]:
    """chunk_sha256 → vector for this version's current chunks; taken *before* a
    structural rewrite deletes them, so unchanged content is not re-embedded."""
    rows = session.execute(
        select(Chunk.chunk_sha256, ChunkEmbedding.embedding)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .where(
            Chunk.version_id == version.id,
            ChunkEmbedding.model_name == provider.model_name,
            ChunkEmbedding.model_version == provider.model_version,
        )
    ).all()
    return {sha: list(vec) for sha, vec in rows}


def embed_version_chunks(
    session: Session,
    version: RegulationVersion,
    provider: EmbeddingProvider,
    *,
    batch_size: int = _BATCH,
    reuse_pool: dict[str, list[float]] | None = None,
) -> EmbedStats:
    chunks = session.scalars(select(Chunk).where(Chunk.version_id == version.id).order_by(Chunk.ordinal)).all()
    already = set(
        session.scalars(
            select(ChunkEmbedding.chunk_id)
            .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
            .where(
                Chunk.version_id == version.id,
                ChunkEmbedding.model_name == provider.model_name,
                ChunkEmbedding.model_version == provider.model_version,
            )
        ).all()
    )
    todo = [c for c in chunks if c.id not in already]
    if not todo:
        return EmbedStats(total=len(chunks), reused=0, embedded=0)

    # Content-addressed reuse across sibling versions of the same regulation.
    sibling_ids = select(RegulationVersion.id).where(RegulationVersion.regulation_id == version.regulation_id)
    wanted_shas = {c.chunk_sha256 for c in todo}
    reusable: dict[str, list[float]] = dict(reuse_pool or {})
    rows = session.execute(
        select(Chunk.chunk_sha256, ChunkEmbedding.embedding)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .where(
            Chunk.version_id.in_(sibling_ids),
            Chunk.chunk_sha256.in_(wanted_shas),
            ChunkEmbedding.model_name == provider.model_name,
            ChunkEmbedding.model_version == provider.model_version,
        )
    ).all()
    for sha, vec in rows:
        reusable.setdefault(sha, list(vec))

    reused = 0
    fresh: list[Chunk] = []
    for c in todo:
        vec = reusable.get(c.chunk_sha256)
        if vec is not None:
            session.add(_row(c.id, provider, vec))
            reused += 1
        else:
            fresh.append(c)

    for i in range(0, len(fresh), batch_size):
        batch = fresh[i : i + batch_size]
        vectors = provider.embed_documents([c.content for c in batch])
        for c, vec in zip(batch, vectors, strict=True):
            session.add(_row(c.id, provider, vec))
    session.flush()
    return EmbedStats(total=len(chunks), reused=reused, embedded=len(fresh))


def _row(chunk_id: uuid.UUID, provider: EmbeddingProvider, vec: list[float]) -> ChunkEmbedding:
    return ChunkEmbedding(
        chunk_id=chunk_id,
        model_name=provider.model_name,
        model_version=provider.model_version,
        dimensions=provider.dimensions,
        embedding=vec,
    )
