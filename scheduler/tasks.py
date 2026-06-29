import os
import sys
from loguru import logger
from database.connection import SessionLocal
from database.models import Regulation, Document, Chunk
from parser.pdf_parser import PDFParser
from registry.metadata_extractor import MetadataExtractor
from registry.version_control import verify_and_register_document
from vectorization.chunker import RegulationChunker
from vectorization.embedder import RegulationEmbedder
from vectorization.incremental_index import prepare_chunks_for_index
from vectorization.indexer import RegulationIndexer
from scheduler.celery_app import celery_app
from app.config import settings

# Lazy singletons for deep learning models to avoid memory issues on celery import
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = RegulationEmbedder()
    return _embedder


@celery_app.task(name="scheduler.tasks.ingest_document_task")
def ingest_document_task(file_path: str, manual_metadata: dict = None) -> dict:
    """
    Executes the ingestion pipeline for a document:
    Extracts metadata -> Verifies version -> Parses PDF & Tables -> Chunks -> Embeds -> Indexes.
    """
    logger.info(f"Starting ingestion task for: {file_path}")
    
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg}

    db = SessionLocal()
    try:
        # 1. Parse document text sample for metadata extraction (first 3 pages)
        parser = PDFParser(file_path)
        pages_sample = parser.extract_metadata()
        page_count = pages_sample.get("page_count", 0)
        
        # Read text of first 3 pages
        sample_text = ""
        try:
            doc_sample = parser.parse()
            for page in doc_sample[:3]:
                sample_text += page["text"] + "\n"
        except Exception as pe:
            logger.error(f"Failed parsing text sample: {pe}")
            
        # 2. Extract metadata
        extractor = MetadataExtractor()
        extracted_metadata = extractor.extract(sample_text, os.path.basename(file_path))

        from registry.amendment_audit import audit_amendment_metadata

        audit = audit_amendment_metadata(extracted_metadata, os.path.basename(file_path))
        extracted_metadata.update(audit)
        
        # Merge with manual metadata if provided (manual overrides)
        if manual_metadata:
            extracted_metadata.update(manual_metadata)
            
        # Convert date strings to python date objects (required by SQLite)
        for df in ["effective_date", "publication_date"]:
            if df in extracted_metadata and isinstance(extracted_metadata[df], str):
                extracted_metadata[df] = MetadataExtractor.parse_date(extracted_metadata[df])
            
        # 3. Version Tracking Check
        regulation, document, state_msg = verify_and_register_document(
            db, file_path, extracted_metadata
        )
        
        if state_msg == "NO_CHANGE":
            logger.info(f"Document content has not changed. Skipping parsing and vectorizing.")
            return {
                "status": "success",
                "state": "NO_CHANGE",
                "regulation_id": regulation.id,
                "document_id": document.id,
                "chunks_count": 0
            }
            
        # Update page count
        document.page_count = page_count
        document.amendment_from_text = extracted_metadata.get("amendment_from_text")
        document.amendment_from_filename = extracted_metadata.get("amendment_from_filename")
        document.amendment_mismatch = bool(extracted_metadata.get("amendment_mismatch"))
        db.add(document)
        db.commit()

        # 4. Parse full document (text + tables)
        parsed_pages = parser.parse()
        
        # 5. Chunk pages (flat by default; structure-aware when explicitly enabled)
        if settings.STRUCTURE_CHUNKING:
            from vectorization.structure_chunker import StructureAwareChunker

            chunker = StructureAwareChunker()
        else:
            chunker = RegulationChunker()
        chunks_data = chunker.chunk_document(parsed_pages, extracted_metadata, document.document_name)

        # 6. Incremental embeddings — reuse unchanged chunk hashes
        embedder = get_embedder()
        chunks_data, embed_stats = prepare_chunks_for_index(
            db, document.id, chunks_data, embedder
        )
        logger.info("Embed stats for %s: %s", document.document_name, embed_stats)

        # 7. Bulk index chunks in DB
        indexer = RegulationIndexer()
        indexed_count = indexer.index_chunks(db, document.id, chunks_data)
        
        logger.info(f"Completed ingestion for {document.document_name}. Status: {state_msg}. Chunks: {indexed_count}")
        return {
            "status": "success",
            "state": state_msg,
            "regulation_id": regulation.id,
            "document_id": document.id,
            "chunks_count": indexed_count
        }
        
    except Exception as e:
        logger.exception(f"Ingestion failed for {file_path}: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()


@celery_app.task(name="scheduler.tasks.run_crawlers_task")
def run_crawlers_task(mock: bool = None, full_fetch: bool = True) -> dict:
    """
    Triggers scrapers and runs the gated acquisition pipeline.
    full_fetch=False performs HEAD/ETag checks and skips unchanged sources.
    """
    from crawler.unece import UNECECrawler
    from crawler.euroncap import EuroNCAPCrawler
    from crawler.nhtsa import NHTSACrawler
    from crawler.iihs import IIHSCrawler
    from registry.pipeline import run_pipeline_for_items

    if mock is None:
        mock = os.getenv("CRAWL_MOCK", "false").lower() == "true"

    logger.info(f"Running regulatory crawlers. mock={mock} full_fetch={full_fetch}")

    crawlers = [
        UNECECrawler(),
        EuroNCAPCrawler(),
        NHTSACrawler(),
        IIHSCrawler(),
    ]

    crawled_items = []
    for cr in crawlers:
        try:
            crawled_items.extend(cr.crawl(mock=mock))
        except Exception as e:
            logger.error(f"Crawler {cr.__class__.__name__} failed: {e}")

    pipeline_result = run_pipeline_for_items(
        crawled_items,
        full_fetch=full_fetch,
        enqueue_ingest=True,
    )

    return {
        "status": "success",
        "mock": mock,
        "full_fetch": full_fetch,
        "crawled_count": len(crawled_items),
        "pipeline": pipeline_result,
    }


@celery_app.task(name="scheduler.tasks.reindex_task")
def reindex_task() -> dict:
    """
    Regenerates vector embeddings for all document chunks in the database.
    Useful when the embedding model is updated.
    """
    logger.info("Starting reindexing task for all chunks in the database.")
    db = SessionLocal()
    try:
        embedder = get_embedder()
        # Fetch chunks in batches to prevent memory blowout
        chunk_query = db.query(Chunk)
        total_chunks = chunk_query.count()
        logger.info(f"Reindexing a total of {total_chunks} chunks.")
        
        batch_size = 100
        processed = 0
        
        for i in range(0, total_chunks, batch_size):
            chunks_batch = chunk_query.offset(i).limit(batch_size).all()
            texts = [c.chunk_text for c in chunks_batch]
            embeddings = embedder.embed_chunks(texts)
            
            for idx, chunk in enumerate(chunks_batch):
                chunk.embedding = embeddings[idx]
                db.add(chunk)
                
            db.commit()
            processed += len(chunks_batch)
            logger.info(f"Reindexed {processed}/{total_chunks} chunks.")
            
        return {"status": "success", "reindexed_count": processed}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Reindexing failed: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()
