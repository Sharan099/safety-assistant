"""
embed_chunks.py — Embed hierarchical chunks (chunks-only, no KG entities).

Uses: BAAI/bge-base-en-v1.5 (config.EMBEDDING_MODEL)
Output: output/regulation_embeddings.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CHUNKS_FILE, EMBEDDING_BATCH, EMBEDDING_MODEL, EMBEDDINGS_FILE  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def p(msg: str) -> None:
    print(msg, flush=True)


def build_chunk_embedding_text(chunk: dict) -> str:
    regulation = chunk.get("regulation", "UNKNOWN")
    heading = chunk.get("heading_path") or chunk.get("section_title", "")
    chunk_type = chunk.get("chunk_type", "paragraph")
    text = chunk.get("text", "")

    tags = []
    if chunk.get("has_test_procedure"):
        tags.append("TEST_PROCEDURE")
    if chunk.get("has_loads"):
        tags.append("LOAD_REQUIREMENTS")
    if chunk.get("has_requirements"):
        tags.append("COMPLIANCE")
    if chunk.get("has_belt_system"):
        tags.append("RESTRAINT")

    return (
        f"[REGULATION {regulation}]\n"
        f"[HEADING {heading}]\n"
        f"[TYPE {chunk_type}]\n"
        f"[TAGS {' '.join(tags)}]\n\n"
        f"{text}"
    ).strip()


def run() -> dict:
    if not CHUNKS_FILE.exists():
        p(f"ERROR: {CHUNKS_FILE} not found. Run hierarchical_chunker first.")
        sys.exit(1)

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    # Embed leaf + section chunks (skip empty)
    to_embed = [c for c in chunks if (c.get("text") or "").strip()]
    p(f"Embedding {len(to_embed)} chunks with {EMBEDDING_MODEL}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL,device="cpu")
    ids = [c["chunk_id"] for c in to_embed]

    t0 = time.perf_counter()
    embeddings: dict[str, list[float]] = {}
    save_every = int(os.getenv("EMBED_SAVE_EVERY", "200"))
    for start in range(0, len(to_embed), EMBEDDING_BATCH):
        batch_chunks = to_embed[start : start + EMBEDDING_BATCH]
        batch_texts = [build_chunk_embedding_text(c) for c in batch_chunks]
        batch_ids = [c["chunk_id"] for c in batch_chunks]
        vectors = model.encode(
            batch_texts,
            batch_size=min(EMBEDDING_BATCH, len(batch_texts)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for cid, vec in zip(batch_ids, vectors):
            embeddings[cid] = vec.tolist()
        if save_every and len(embeddings) % save_every < EMBEDDING_BATCH:
            gc.collect()
    metadata = {
        cid: {
            "type": c.get("chunk_type"),
            "regulation": c.get("regulation"),
            "heading_path": c.get("heading_path"),
            "parent_id": c.get("parent_id"),
        }
        for cid, c in zip(ids, to_embed)
    }

    out = {
        "model": EMBEDDING_MODEL,
        "dimension": getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)(),
        "total_vectors": len(embeddings),
        "embeddings": embeddings,
        "metadata": metadata,
    }

    EMBEDDINGS_FILE.write_text(
        json.dumps(out, ensure_ascii=False),
        encoding="utf-8",
    )
    elapsed = round(time.perf_counter() - t0, 1)
    p(f"Saved {len(embeddings)} vectors -> {EMBEDDINGS_FILE} ({elapsed}s)")
    return out


if __name__ == "__main__":
    run()
