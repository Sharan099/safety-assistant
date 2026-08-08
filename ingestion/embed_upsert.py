"""Embed enriched chunks and upsert into Qdrant (dense + BM25 sparse)."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ingestion.enrich import iter_embed_texts
from ingestion.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "regulations"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH = 64
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
BM25_MODEL = "Qdrant/bm25"


def _point_uuid(chunk_id: str, regulation_id: str) -> str:
    """Stable UUIDv5 so re-ingest of the same chunk overwrites cleanly."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{regulation_id}:{chunk_id}"))


class Embedder:
    """Thin wrapper around sentence-transformers (dense)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            from ingestion.hf_auth import ensure_hf_auth

            ensure_hf_auth()
            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str], *, batch_size: int = DEFAULT_BATCH) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        try:
            from observability.context import get_current_trace

            tr = get_current_trace()
            if tr is not None:
                # Word-count proxy when tokenizer usage is unavailable.
                tokens = sum(max(1, len(t.split())) for t in texts)
                tr.add_embedding(model=self.model_name, tokens=tokens)
        except Exception:  # noqa: BLE001
            pass
        return [v.tolist() for v in vectors]


def get_qdrant_client(
    *,
    url: str | None = None,
    api_key: str | None = None,
    path: str | None = None,
) -> QdrantClient:
    load_dotenv()
    url = url or os.getenv("QDRANT_URL")
    api_key = api_key or os.getenv("QDRANT_API_KEY")
    path = path or os.getenv("QDRANT_PATH", "./data/qdrant")

    if url:
        logger.info("Connecting to Qdrant at %s", url)
        return QdrantClient(url=url, api_key=api_key)

    Path(path).mkdir(parents=True, exist_ok=True)
    logger.info("Using local Qdrant path %s", path)
    return QdrantClient(path=path)


def _is_hybrid_collection(info: Any, *, vector_size: int) -> bool:
    """True when collection has named dense + bm25 sparse of the expected dim."""
    params = info.config.params
    vectors = params.vectors
    sparse = getattr(params, "sparse_vectors", None) or {}

    if not isinstance(vectors, dict):
        return False
    dense = vectors.get(DENSE_VECTOR)
    if dense is None or getattr(dense, "size", None) != vector_size:
        return False
    return SPARSE_VECTOR in sparse


def ensure_collection(
    client: QdrantClient,
    *,
    collection: str = DEFAULT_COLLECTION,
    vector_size: int,
) -> None:
    """Create (or recreate) a hybrid collection: named dense + BM25 sparse."""
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        info = client.get_collection(collection)
        if _is_hybrid_collection(info, vector_size=vector_size):
            return
        logger.warning(
            "Collection %s is not hybrid dense+bm25 (dim=%s) — recreating",
            collection,
            vector_size,
        )
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR: qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR: qm.SparseVectorParams(modifier=qm.Modifier.IDF),
        },
    )
    # Metadata contract + legacy retrieval indexes (see ingestion.wipe.METADATA_CONTRACT_FIELDS).
    for field_name, schema in (
        ("document_id", qm.PayloadSchemaType.KEYWORD),
        ("page_number", qm.PayloadSchemaType.INTEGER),
        ("section", qm.PayloadSchemaType.KEYWORD),
        ("section_number", qm.PayloadSchemaType.KEYWORD),
        ("element_type", qm.PayloadSchemaType.KEYWORD),
        ("parent_id", qm.PayloadSchemaType.KEYWORD),
        ("element_id", qm.PayloadSchemaType.KEYWORD),
        ("regulation_id", qm.PayloadSchemaType.KEYWORD),
        ("revision", qm.PayloadSchemaType.KEYWORD),
        ("section_id", qm.PayloadSchemaType.KEYWORD),
        ("parent_section_id", qm.PayloadSchemaType.KEYWORD),
        ("content_type", qm.PayloadSchemaType.KEYWORD),
        ("chunk_id", qm.PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception:  # noqa: BLE001 — index may already exist
            pass


def delete_regulation(
    client: QdrantClient,
    *,
    regulation_id: str,
    collection: str = DEFAULT_COLLECTION,
) -> None:
    """Idempotent prep: remove all points for this regulation_id."""
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        return
    logger.info("Deleting existing points for regulation_id=%s", regulation_id)
    client.delete(
        collection_name=collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="regulation_id",
                        match=qm.MatchValue(value=regulation_id),
                    )
                ]
            )
        ),
    )


def upsert_chunks(
    chunks: Sequence[Chunk],
    *,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
    collection: str | None = None,
    batch_size: int = DEFAULT_BATCH,
    delete_existing: bool = True,
) -> int:
    """Embed + upsert chunks with dense + BM25 vectors."""
    load_dotenv()
    if not chunks:
        return 0

    collection = collection or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    client = client or get_qdrant_client()
    embedder = embedder or Embedder()

    ensure_collection(client, collection=collection, vector_size=embedder.dim)

    regulation_id = chunks[0].regulation_id
    if delete_existing:
        delete_regulation(client, regulation_id=regulation_id, collection=collection)

    texts = iter_embed_texts(chunks)
    logger.info("Embedding %d chunks with %s (+ BM25)", len(texts), embedder.model_name)
    vectors = embedder.embed(texts, batch_size=batch_size)

    points: list[qm.PointStruct] = []
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for chunk, dense, text in zip(chunks, vectors, texts, strict=True):
        payload = chunk.payload()
        payload["ingested_at"] = chunk.ingested_at or ingested_at
        points.append(
            qm.PointStruct(
                id=_point_uuid(chunk.chunk_id, chunk.regulation_id),
                vector={
                    DENSE_VECTOR: dense,
                    SPARSE_VECTOR: qm.Document(text=text, model=BM25_MODEL),
                },
                payload=payload,
            )
        )

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection, points=batch)
        logger.info("Upserted %d–%d / %d", i + 1, i + len(batch), len(points))

    try:
        from retrieval.retrieve import invalidate_indexed_regulations_cache

        invalidate_indexed_regulations_cache()
    except Exception:  # noqa: BLE001
        pass
    return len(points)
