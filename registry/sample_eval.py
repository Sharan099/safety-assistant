"""Retrieval eval for sample flat vs structure-aware staging indices."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from registry.embedding_config import EMBEDDING_DIMENSION
from registry.staging_index import CHUNKS_FILE, EMBEDDINGS_FILE, SAMPLE_DOCUMENTS
from vectorization.embedder import RegulationEmbedder

SAMPLE_CASES_PATH = Path(__file__).resolve().parents[1] / "tests" / "data" / "sample_retrieval_cases.json"


def _tokenize(text: str) -> list[str]:
    return [t for t in re.sub(r"[^\w\s]", " ", text.lower()).split() if len(t) > 2]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_staging_chunks() -> tuple[list[dict], dict[str, list[float]]]:
    chunks_data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    emb_data = json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))
    return chunks_data["chunks"], emb_data["embeddings"]


def _flat_sample_chunks(db: object | None = None) -> list[dict]:
    """Load flat baseline chunks for the three sample PDFs (ORM-free for unmigrated SQLite)."""
    import sqlite3

    from database.connection import engine

    db_path = str(engine.url).replace("sqlite:///", "")
    if not Path(db_path).exists():
        return []

    sample_names = {Path(spec["file_path"]).name for spec in SAMPLE_DOCUMENTS.values()}
    codes = [spec["metadata"]["regulation_code"] for spec in SAMPLE_DOCUMENTS.values()]
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in sample_names)
    code_ph = ",".join("?" for _ in codes)
    query = f"""
        SELECT ch.id, ch.chunk_text, ch.embedding, r.regulation_code, d.document_name
        FROM chunks ch
        JOIN documents d ON ch.document_id = d.id
        JOIN regulations r ON d.regulation_id = r.id
        WHERE d.document_name IN ({placeholders})
          AND r.regulation_code IN ({code_ph})
    """
    rows = conn.execute(query, (*sample_names, *codes)).fetchall()
    conn.close()

    out: list[dict] = []
    for cid, text, emb_json, reg_code, doc_name in rows:
        emb = json.loads(emb_json) if emb_json else None
        out.append(
            {
                "chunk_id": str(cid),
                "chunk_text": text,
                "regulation_code": reg_code,
                "embedding": emb,
                "chunk_type": "flat",
                "heading_path": doc_name,
            }
        )
    return out


def _hybrid_search(
    query: str,
    chunks: list[dict],
    embeddings: dict[str, list[float]] | None,
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
        cid = str(ch["chunk_id"])
        if embeddings and cid in embeddings:
            vec = np.array(embeddings[cid], dtype=float)
        elif ch.get("embedding"):
            vec = np.array(ch["embedding"], dtype=float)
        else:
            continue
        dense = _cosine(q_vec, vec)
        text = ch.get("chunk_text", "").lower()
        sparse = sum(1 for t in q_terms if t in text) / max(len(q_terms), 1)
        score = 0.6 * dense + 0.4 * sparse
        scored.append((score, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def _chunk_contains_all_terms(chunk_text: str, terms: list[str]) -> bool:
    lower = chunk_text.lower()
    for term in terms:
        t = term.lower()
        if t not in lower:
            return False
    return True


def _table_integrity_chunk(chunk: dict, *, min_pipes: int) -> bool:
    text = chunk.get("chunk_text", "")
    if chunk.get("chunk_type") == "table":
        return text.count("|") >= min_pipes
    return text.count("|") >= min_pipes and "| --- |" in text


def _case_pass(case: dict, hits: list[dict]) -> dict[str, Any]:
    must_contain = [s.lower() for s in case.get("must_contain", [])]
    must_not = [s.lower() for s in case.get("must_not_contain", [])]
    expected_regs = case.get("expected_regs", [])
    together = case.get("must_contain_together", [])
    min_pipes = int(case.get("min_table_pipes", 20))
    require_table = bool(case.get("require_single_chunk_table"))

    found_regs = {h.get("regulation_code") for h in hits}
    texts = " ".join(h.get("chunk_text", "") for h in hits).lower()

    contains_ok = all(term in texts for term in must_contain) if must_contain else True
    forbidden_hit = []
    for term in must_not:
        if term.startswith(" ") or term.endswith(" "):
            if term in texts:
                forbidden_hit.append(term.strip())
        elif re.search(rf"\b{re.escape(term)}\b", texts):
            forbidden_hit.append(term)

    reg_ok = all(reg in found_regs for reg in expected_regs) if expected_regs else bool(hits)

    together_chunk_id = None
    together_ok = True
    if together:
        together_ok = False
        for hit in hits:
            if _chunk_contains_all_terms(hit.get("chunk_text", ""), together):
                together_ok = True
                together_chunk_id = str(hit.get("chunk_id"))
                break

    table_chunk_id = None
    table_ok = True
    if require_table:
        table_ok = False
        for hit in hits:
            if _table_integrity_chunk(hit, min_pipes=min_pipes) and (
                not together or _chunk_contains_all_terms(hit.get("chunk_text", ""), together)
            ):
                table_ok = True
                table_chunk_id = str(hit.get("chunk_id"))
                break

    passed = (
        contains_ok
        and reg_ok
        and not forbidden_hit
        and together_ok
        and table_ok
    )
    return {
        "id": case["id"],
        "passed": passed,
        "contains_ok": contains_ok,
        "reg_ok": reg_ok,
        "together_ok": together_ok,
        "table_ok": table_ok,
        "forbidden_hit": forbidden_hit,
        "together_chunk_id": together_chunk_id,
        "table_chunk_id": table_chunk_id,
        "top_ids": [str(h.get("chunk_id")) for h in hits[:5]],
        "retrieved": len(hits),
    }


def _compare_case(case: dict, flat: dict, struct: dict) -> str:
    if struct["passed"] and not flat["passed"]:
        return "structure_wins"
    if flat["passed"] and not struct["passed"]:
        return "flat_wins"
    if flat["passed"] and struct["passed"]:
        return "tie"
    return "both_fail"


def run_sample_eval(db: object | None = None, *, top_k: int = 8) -> dict[str, Any]:
    cases = json.loads(SAMPLE_CASES_PATH.read_text(encoding="utf-8"))
    embedder = RegulationEmbedder()

    flat_chunks = _flat_sample_chunks(db)
    struct_chunks, struct_emb = _load_staging_chunks()

    per_case: list[dict[str, Any]] = []
    flat_results = []
    struct_results = []
    for case in cases:
        reg = case.get("expected_regs", [None])[0]
        flat_hits = _hybrid_search(
            case["query"], flat_chunks, None, embedder, top_k=top_k, regulation_filter=reg
        )
        struct_hits = _hybrid_search(
            case["query"], struct_chunks, struct_emb, embedder, top_k=top_k, regulation_filter=reg
        )
        flat_r = _case_pass(case, flat_hits)
        struct_r = _case_pass(case, struct_hits)
        flat_results.append(flat_r)
        struct_results.append(struct_r)
        outcome = _compare_case(case, flat_r, struct_r)
        per_case.append(
            {
                "id": case["id"],
                "structure_should_win": case.get("structure_should_win", False),
                "flat": flat_r,
                "structure": struct_r,
                "outcome": outcome,
            }
        )

    def _summary(results: list[dict]) -> dict[str, Any]:
        passed = sum(1 for r in results if r["passed"])
        return {
            "passed": passed,
            "total": len(results),
            "recall_rate": round(passed / len(results), 3) if results else 0.0,
            "results": results,
        }

    flat = _summary(flat_results)
    struct = _summary(struct_results)
    structure_wins = sum(1 for c in per_case if c["outcome"] == "structure_wins")
    flat_wins = sum(1 for c in per_case if c["outcome"] == "flat_wins")
    ties = sum(1 for c in per_case if c["outcome"] == "tie")
    regressions = [c["id"] for c in per_case if c["outcome"] == "flat_wins"]
    improvements = [c["id"] for c in per_case if c["outcome"] == "structure_wins"]

    return {
        "top_k": top_k,
        "per_case": per_case,
        "flat_baseline": flat,
        "structure_staging": struct,
        "summary": {
            "structure_wins": structure_wins,
            "flat_wins": flat_wins,
            "ties": ties,
            "regressions": regressions,
            "improvements": improvements,
            "approval_ready": flat_wins == 0 and struct["passed"] == struct["total"],
        },
        "delta_passed": struct["passed"] - flat["passed"],
        "improved_cases": improvements,
        "regressed_cases": regressions,
    }
