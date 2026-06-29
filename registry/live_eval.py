"""Spot-check hard retrieval cases against the live indexed corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from database.models import Chunk, Document, Regulation
from registry.embedding_config import EMBEDDING_DIMENSION
from registry.sample_eval import SAMPLE_CASES_PATH, _case_pass, _tokenize
from vectorization.embedder import RegulationEmbedder

CASES_PATH = SAMPLE_CASES_PATH


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_live_chunks(db: Session) -> list[dict]:
    rows = (
        db.query(
            Chunk.id,
            Chunk.chunk_text,
            Chunk.embedding,
            Chunk.chunk_type,
            Chunk.heading_path,
            Chunk.parent_chunk_id,
            Regulation.regulation_code,
        )
        .join(Document, Chunk.document_id == Document.id)
        .join(Regulation, Document.regulation_id == Regulation.id)
        .filter(Regulation.status == "ACTIVE")
        .all()
    )
    out: list[dict] = []
    for cid, text, emb, ctype, heading, parent_id, reg_code in rows:
        vec = emb
        if isinstance(emb, str):
            vec = json.loads(emb)
        out.append(
            {
                "chunk_id": str(cid),
                "chunk_text": text,
                "embedding": vec,
                "chunk_type": ctype,
                "heading_path": heading,
                "parent_chunk_id": parent_id,
                "regulation_code": reg_code,
            }
        )
    return out


def _hybrid_search_live(
    query: str,
    chunks: list[dict],
    embedder: RegulationEmbedder,
    *,
    top_k: int = 8,
    regulation_filter: str | None = None,
) -> list[dict]:
    if regulation_filter:
        chunks = [c for c in chunks if c.get("regulation_code") == regulation_filter]
    if not chunks:
        return []

    q_vec = np.array(embedder.embed_query(query), dtype=float)
    if q_vec.shape[0] != EMBEDDING_DIMENSION:
        raise ValueError(f"Query embedding dimension {q_vec.shape[0]} != {EMBEDDING_DIMENSION}")

    q_terms = set(_tokenize(query))
    scored: list[tuple[float, dict]] = []
    for ch in chunks:
        vec = ch.get("embedding")
        if not vec:
            continue
        dense = _cosine(q_vec, np.array(vec, dtype=float))
        text = ch.get("chunk_text", "").lower()
        sparse = sum(1 for t in q_terms if t in text) / max(len(q_terms), 1)
        scored.append((0.6 * dense + 0.4 * sparse, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def run_live_spot_check(db: Session, *, top_k: int = 8) -> dict[str, Any]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    embedder = RegulationEmbedder()
    chunks = _load_live_chunks(db)

    null_parent_reads = sum(1 for c in chunks if c.get("parent_chunk_id") is None)
    with_parent = len(chunks) - null_parent_reads

    results: list[dict[str, Any]] = []
    for case in cases:
        reg = case.get("expected_regs", [None])[0]
        hits = _hybrid_search_live(
            case["query"], chunks, embedder, top_k=top_k, regulation_filter=reg
        )
        case_result = _case_pass(case, hits)
        results.append(
            {
                "id": case["id"],
                "passed": case_result["passed"],
                "together_chunk_id": case_result.get("together_chunk_id"),
                "table_chunk_id": case_result.get("table_chunk_id"),
                "top_ids": case_result.get("top_ids"),
                "detail": case_result,
            }
        )

    passed = sum(1 for r in results if r["passed"])
    return {
        "top_k": top_k,
        "embedding_mock": embedder.use_mock_embeddings,
        "embedding_backend": embedder.backend,
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "chunk_count": len(chunks),
        "parent_chunk_id_null_safe": null_parent_reads,
        "parent_chunk_id_set": with_parent,
        "passed": passed,
        "total": len(results),
        "all_passed": passed == len(results),
        "results": results,
    }
