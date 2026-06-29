"""Staging index for sample structure-aware chunking (no live corpus re-chunk)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from parser.pdf_parser import PDFParser
from registry.embedding_config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from vectorization.embedder import RegulationEmbedder
from vectorization.structure_chunker import StructureAwareChunker

STAGING_DIR = Path("output/staging_index")
CHUNKS_FILE = STAGING_DIR / "sample_chunks.json"
EMBEDDINGS_FILE = STAGING_DIR / "sample_embeddings.json"

SAMPLE_DOCUMENTS: dict[str, dict[str, Any]] = {
    "UN_R94": {
        "file_path": "storage/UNECE/UN_R94.pdf",
        "metadata": {
            "regulation_code": "UN_R94",
            "source_type": "UNECE",
            "amendment": "04 Series",
            "title": "UN Regulation No. 94",
        },
    },
    "UN_R16": {
        "file_path": "storage/UNECE/UN_R16_08Series.pdf",
        "metadata": {
            "regulation_code": "UN_R16",
            "source_type": "UNECE",
            "amendment": "08 Series",
            "title": "UN Regulation No. 16",
        },
    },
    "UN_R14": {
        "file_path": "storage/UNECE/UN_R14_07Series.pdf",
        "metadata": {
            "regulation_code": "UN_R14",
            "source_type": "UNECE",
            "amendment": "07 Series",
            "title": "UN Regulation No. 14",
        },
    },
}


def build_staging_index(root: Path | None = None) -> dict[str, Any]:
    root = root or Path.cwd()
    chunker = StructureAwareChunker()
    embedder = RegulationEmbedder()
    all_chunks: list[dict[str, Any]] = []
    embeddings: dict[str, list[float]] = {}

    for key, spec in SAMPLE_DOCUMENTS.items():
        pdf_path = root / spec["file_path"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Sample document missing: {pdf_path}")
        parser = PDFParser(str(pdf_path))
        pages = parser.parse()
        doc_name = pdf_path.name
        chunks = chunker.chunk_document(pages, spec["metadata"], doc_name)
        texts = [c["chunk_text"] for c in chunks]
        vectors = embedder.embed_chunks(texts)
        for idx, ch in enumerate(chunks):
            chunk_id = f"{key}_{ch['chunk_index']}"
            ch["chunk_id"] = chunk_id
            ch["regulation_code"] = spec["metadata"]["regulation_code"]
            ch["embedding"] = vectors[idx]
            all_chunks.append(ch)
            embeddings[chunk_id] = vectors[idx]

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    chunks_payload = {
        "index_type": "structure_aware_sample",
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
        "documents": list(SAMPLE_DOCUMENTS.keys()),
        "total_chunks": len(all_chunks),
        "chunks": [{k: v for k, v in c.items() if k != "embedding"} for c in all_chunks],
    }
    emb_payload = {
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
        "total_vectors": len(embeddings),
        "embeddings": embeddings,
    }
    CHUNKS_FILE.write_text(json.dumps(chunks_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    EMBEDDINGS_FILE.write_text(json.dumps(emb_payload, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote staging index: %s chunks -> %s", len(all_chunks), STAGING_DIR)
    return chunks_payload
