from typing import List, Dict, Any
import json

from sqlalchemy.orm import Session
from database.models import Chunk
from loguru import logger


def delete_chunks_for_document(db: Session, document_id: int) -> int:
    """Remove all chunks for a document (idempotent re-index / supersede-safe replace)."""
    deleted = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .delete(synchronize_session=False)
    )
    if deleted:
        db.commit()
        logger.info(f"Deleted {deleted} existing chunks for document_id {document_id}")
    return deleted


class RegulationIndexer:
    """Bulk indexing of chunks with structure metadata and parent-child links."""

    def index_chunks(self, db: Session, document_id: int, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            logger.warning(f"No chunks to index for document_id {document_id}")
            return 0

        try:
            delete_chunks_for_document(db, document_id)
            pending_parents: list[tuple[Chunk, int | None]] = []
            for item in chunks:
                chunk_obj = Chunk(
                    document_id=document_id,
                    chunk_text=item["chunk_text"],
                    chunk_index=item["chunk_index"],
                    page_number=item.get("page_number"),
                    section=item.get("section"),
                    paragraph=item.get("paragraph"),
                    chunk_type=item.get("chunk_type"),
                    heading_path=item.get("heading_path"),
                    content_hash=item.get("content_hash"),
                    provenance=json.dumps(item.get("metadata", {})) if item.get("metadata") else None,
                    embedding=item.get("embedding"),
                )
                db.add(chunk_obj)
                pending_parents.append((chunk_obj, item.get("parent_chunk_index")))
            db.flush()

            index_to_id = {ch.chunk_index: ch.id for ch, _ in pending_parents}
            for chunk_obj, parent_idx in pending_parents:
                if parent_idx is not None and parent_idx in index_to_id:
                    chunk_obj.parent_chunk_id = index_to_id[parent_idx]

            db.commit()
            logger.info(
                "Successfully indexed %s chunks for document %s",
                len(pending_parents),
                document_id,
            )
            return len(pending_parents)
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving chunks for document {document_id}: {e}")
            raise
