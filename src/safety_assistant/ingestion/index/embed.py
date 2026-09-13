"""Embedding/indexing stage with content-addressed reuse.

``embed  = text_sha256 + embedding_model_version + dimensions + representation``

Two representations of a chunk can be indexed side by side:

- ``content``  the baseline: the chunk text itself;
- ``sac_v1``   summary-augmented: ``chunks.retrieval_text`` (identity block + summary + content).

Before embedding a chunk we look for an existing embedding of *the same text* (same
sha256) under *any* version of the same regulation with the same model — an amendment that
leaves 95 % of clauses untouched re-embeds 5 %. Reuse is by value copy, so every chunk row
still owns one embedding row per representation and cascades cleanly. SAC text carries the
version identity, so its cross-version reuse is nil by design; within-version reuse (a
structural re-chunk) still applies through ``reuse_pool``.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from safety_assistant.persistence.models import Chunk, ChunkEmbedding, RegulationVersion
from safety_assistant.providers.embeddings import EmbeddingProvider

INDEX_SCHEMA_VERSION = 1  # unchanged by SAC: the sac_v1 representation is an additional index, built by `reindex`
_BATCH = 64
CONTENT = "content"
SAC = "sac_v1"
SAC_COMPACT = "sac_v2"


@dataclass
class EmbedStats:
    total: int
    reused: int
    embedded: int


def _text(chunk: Chunk, representation: str, prefix: str | None = None) -> str:
    if representation == SAC:
        return chunk.retrieval_text or chunk.content
    if representation == SAC_COMPACT:
        return f"{prefix}\n{chunk.content}" if prefix else chunk.content
    return chunk.content


def _sha(chunk: Chunk, representation: str, prefix: str | None = None) -> str:
    if representation == CONTENT:
        return chunk.chunk_sha256
    return hashlib.sha256(_text(chunk, representation, prefix).encode("utf-8")).hexdigest()


def snapshot_embeddings(
    session: Session, version: RegulationVersion, provider: EmbeddingProvider
) -> dict[str, list[float]]:
    """text sha256 → vector for this version's current chunks (both representations); taken
    *before* a structural rewrite deletes them, so unchanged text is not re-embedded."""
    rows = session.execute(
        select(Chunk, ChunkEmbedding.embedding, ChunkEmbedding.representation)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .where(
            Chunk.version_id == version.id,
            ChunkEmbedding.model_name == provider.model_name,
            ChunkEmbedding.model_version == provider.model_version,
        )
    ).all()
    return {_sha(chunk, rep): list(vec) for chunk, vec, rep in rows if rep != SAC_COMPACT}


def embed_version_chunks(
    session: Session,
    version: RegulationVersion,
    provider: EmbeddingProvider,
    *,
    representation: str = CONTENT,
    batch_size: int = _BATCH,
    reuse_pool: dict[str, list[float]] | None = None,
    prefix: str | None = None,  # sac_v2: the version's compact identity line
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
                ChunkEmbedding.representation == representation,
            )
        ).all()
    )
    todo = [c for c in chunks if c.id not in already]
    if not todo:
        return EmbedStats(total=len(chunks), reused=0, embedded=0)

    reusable: dict[str, list[float]] = dict(reuse_pool or {})
    if representation == CONTENT:
        # Content-addressed reuse across sibling versions of the same regulation.
        sibling_ids = select(RegulationVersion.id).where(RegulationVersion.regulation_id == version.regulation_id)
        wanted_shas = {c.chunk_sha256 for c in todo}
        rows = session.execute(
            select(Chunk.chunk_sha256, ChunkEmbedding.embedding)
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
            .where(
                Chunk.version_id.in_(sibling_ids),
                Chunk.chunk_sha256.in_(wanted_shas),
                ChunkEmbedding.model_name == provider.model_name,
                ChunkEmbedding.model_version == provider.model_version,
                ChunkEmbedding.representation == CONTENT,
            )
        ).all()
        for sha, vec in rows:
            reusable.setdefault(sha, list(vec))

    reused = 0
    fresh: list[Chunk] = []
    for c in todo:
        vec = reusable.get(_sha(c, representation, prefix))
        if vec is not None:
            session.add(_row(c.id, provider, vec, representation))
            reused += 1
        else:
            fresh.append(c)

    for i in range(0, len(fresh), batch_size):
        batch = fresh[i : i + batch_size]
        vectors = provider.embed_documents([_text(c, representation, prefix) for c in batch])
        for c, vec in zip(batch, vectors, strict=True):
            session.add(_row(c.id, provider, vec, representation))
    session.flush()
    return EmbedStats(total=len(chunks), reused=reused, embedded=len(fresh))


def _row(chunk_id: uuid.UUID, provider: EmbeddingProvider, vec: list[float], representation: str) -> ChunkEmbedding:
    return ChunkEmbedding(
        chunk_id=chunk_id,
        model_name=provider.model_name,
        model_version=provider.model_version,
        dimensions=provider.dimensions,
        embedding=vec,
        representation=representation,
    )
