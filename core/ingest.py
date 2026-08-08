"""Incremental ingest for the UNECE corpus — PDF, Markdown, TXT."""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger
from sqlalchemy.orm import Session

from app.config import settings
from core.chunking import adaptive_chunk_document
from core.documents import document_type_for
from core.embedder import EMBEDDING_DIMENSION, Embedder
from core.sources import SOURCES, pdf_dir
from database.connection import SessionLocal, engine
from database.models import Base, Chunk, Document, Regulation


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _header(meta: dict, document_name: str, page: int, section: str) -> str:
    return (
        f"[Source: UNECE | Reg: {meta['regulation_code']} | Doc: {document_name} | "
        f"Page: {page} | Section: {section}]\n\n"
    )


def _find_source_for_file(path: Path) -> dict | None:
    for src in SOURCES:
        if path.name == src["filename"]:
            return src
    return None


def _discover_files() -> list[tuple[dict, Path]]:
    """Pinned PDFs plus optional extra docs in data/docs/."""
    found: list[tuple[dict, Path]] = []
    for src in SOURCES:
        path = pdf_dir() / src["filename"]
        if path.is_file():
            found.append((src, path))

    for path in sorted(settings.DOCS_DIR.glob("*")):
        if path.suffix.lower() not in settings.supported_extensions:
            continue
        src = _find_source_for_file(path)
        if src and any(p == path for _, p in found):
            continue
        if src:
            found.append((src, path))
    return found


def ingest_if_empty() -> dict | None:
    db = SessionLocal()
    try:
        if db.query(Chunk).count() > 0:
            logger.info("Corpus already indexed — skipping ingest")
            return None
    finally:
        db.close()
    return ingest_all(reset=False, incremental=False)


def _document_unchanged(db: Session, reg_id: int, file_hash: str) -> bool:
    doc = (
        db.query(Document)
        .filter(Document.regulation_id == reg_id, Document.hash == file_hash)
        .first()
    )
    return doc is not None and db.query(Chunk).filter(Chunk.document_id == doc.id).count() > 0


def ingest_all(*, reset: bool = False, incremental: bool | None = None) -> dict:
    incremental = settings.INCREMENTAL_INGEST if incremental is None else incremental
    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()
    embedder = Embedder()
    stats = {"documents": 0, "chunks": 0, "skipped": 0, "errors": []}

    try:
        if reset:
            logger.info("Resetting corpus tables")
            db.query(Chunk).delete()
            db.query(Document).delete()
            db.query(Regulation).delete()
            db.commit()

        for src, path in _discover_files():
            file_hash = _sha256(path)
            reg = db.query(Regulation).filter_by(regulation_code=src["regulation_code"]).first()

            if incremental and reg and _document_unchanged(db, reg.id, file_hash):
                stats["skipped"] += 1
                logger.info("Skipping unchanged %s", src["regulation_code"])
                continue

            if reg:
                doc_ids = [
                    d.id for d in db.query(Document).filter(Document.regulation_id == reg.id).all()
                ]
                if doc_ids:
                    db.query(Chunk).filter(Chunk.document_id.in_(doc_ids)).delete(
                        synchronize_session=False
                    )
                db.query(Document).filter(Document.regulation_id == reg.id).delete()
                db.delete(reg)
                db.flush()

            reg = Regulation(
                regulation_code=src["regulation_code"],
                title=src["title"],
                source_type="UNECE",
                amendment="Base",
                status="ACTIVE",
                market="GLOBAL",
                checksum=file_hash,
                local_file_path=str(path),
            )
            db.add(reg)
            db.flush()

            doc = Document(
                regulation_id=reg.id,
                document_name=path.name,
                document_type=document_type_for(path),
                file_path=str(path),
                hash=file_hash,
            )
            db.add(doc)
            db.flush()

            raw_chunks, strategy = adaptive_chunk_document(path)
            texts = [
                _header(src, path.name, c["page_number"], c["section"]) + c["chunk_text"]
                for c in raw_chunks
            ]
            vectors = embedder.embed_passages(texts)

            for raw, text, vector in zip(raw_chunks, texts, vectors):
                db.add(
                    Chunk(
                        document_id=doc.id,
                        chunk_text=text,
                        chunk_index=raw["chunk_index"],
                        page_number=raw["page_number"],
                        section=raw["section"],
                        chunk_type=raw.get("chunk_type", "adaptive"),
                        embedding=vector,
                    )
                )

            stats["documents"] += 1
            stats["chunks"] += len(raw_chunks)
            logger.info(
                "Ingested %s via %s (%d chunks)",
                src["regulation_code"],
                strategy,
                len(raw_chunks),
            )

        db.commit()
        stats["embedding_dimension"] = EMBEDDING_DIMENSION
        return stats
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    result = ingest_all(reset=False, incremental=True)
    print(result)
