"""index_chunks() — embed every DocumentChunk that doesn't have an
Embedding row yet for the given provider. Idempotent per (chunk, model_name,
model_version) — matches DocumentRevision's idempotency in packages/ingestion.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from packages.domain.knowledge import DocumentChunk, Embedding
from packages.retrieval.embeddings import EmbeddingProvider, get_default_embedding_provider


def index_chunks(
    session: Session,
    provider: EmbeddingProvider | None = None,
    *,
    document_revision_id: object | None = None,
) -> int:
    provider = provider or get_default_embedding_provider()  # real semantic embeddings — docs/ADR/0014

    query = session.query(DocumentChunk)
    if document_revision_id is not None:
        query = query.filter_by(document_revision_id=document_revision_id)

    already_embedded = {
        row[0]
        for row in session.query(Embedding.chunk_id)
        .filter(Embedding.model_name == provider.model_name, Embedding.model_version == provider.model_version)
        .all()
    }

    count = 0
    for chunk in query.all():
        if chunk.id in already_embedded:
            continue
        vector = provider.embed(chunk.content)
        session.add(
            Embedding(
                chunk_id=chunk.id,
                model_name=provider.model_name,
                model_version=provider.model_version,
                dimensions=provider.dimensions,
                embedding=vector,
            )
        )
        count += 1
    session.commit()
    return count
