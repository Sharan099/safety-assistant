"""
embed_chunks.py — Embed hierarchical chunks (chunks-only, no KG entities).

Uses: nomic-ai/nomic-embed-text-v1.5 (config.EMBEDDING_MODEL)
Passages are prefixed with EMBEDDING_DOC_PREFIX ("search_document: ") as Nomic
requires; queries are prefixed with "search_query: " at retrieval time.
Output: output/regulation_embeddings.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CHUNKS_FILE,
    EMBEDDING_BATCH,
    EMBEDDING_DIMENSION,
    EMBEDDING_DOC_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_TRUST_REMOTE_CODE,
    EMBEDDINGS_FILE,
)

sys.stdout.reconfigure(line_buffering=True)


def p(msg: str) -> None:
    print(msg, flush=True)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    p(f"[{_ts()}] {msg}")


def _embedding_dimension(model) -> int:
    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return dim_fn()


def _save_checkpoint(
    embeddings: dict[str, list[float]],
    metadata: dict[str, dict],
    model,
    out_path: Path,
    *,
    reason: str = "checkpoint",
) -> None:
    out = {
        "model": EMBEDDING_MODEL,
        "dimension": _embedding_dimension(model),
        "total_vectors": len(embeddings),
        "embeddings": embeddings,
        "metadata": metadata,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    _log(f"Checkpoint saved ({reason}): {len(embeddings)} vectors -> {out_path}")


def _embedding_prefixes() -> tuple[str, str]:
    """Query/document prefix discipline per model family."""
    model = EMBEDDING_MODEL.lower()
    if "nomic" in model:
        return EMBEDDING_QUERY_PREFIX, EMBEDDING_DOC_PREFIX
    if "bge" in model:
        # BGE-M3: instruction-style prefixes for retrieval
        return "search_query: ", "search_document: "
    return EMBEDDING_QUERY_PREFIX, EMBEDDING_DOC_PREFIX


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


def embed_chunks_to_file(chunks: list[dict], out_path: Path) -> dict:
    """Embed an in-memory chunk list and write vectors to out_path."""
    to_embed = [c for c in chunks if (c.get("text") or "").strip()]
    p(f"Embedding {len(to_embed)} chunks -> {out_path} ({EMBEDDING_MODEL})")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
        trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE,
    )
    ids = [c["chunk_id"] for c in to_embed]

    t0 = time.perf_counter()
    embeddings: dict[str, list[float]] = {}
    for start in range(0, len(to_embed), EMBEDDING_BATCH):
        batch_chunks = to_embed[start : start + EMBEDDING_BATCH]
        batch_texts = [
            _embedding_prefixes()[1] + build_chunk_embedding_text(c) for c in batch_chunks
        ]
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
    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    out = {
        "model": EMBEDDING_MODEL,
        "dimension": dim_fn(),
        "total_vectors": len(embeddings),
        "embeddings": embeddings,
        "metadata": metadata,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    elapsed = round(time.perf_counter() - t0, 1)
    p(f"Saved {len(embeddings)} vectors ({elapsed}s)")
    return out


def run() -> dict:
    if not CHUNKS_FILE.exists():
        p(f"ERROR: {CHUNKS_FILE} not found. Run hierarchical_chunker first.")
        sys.exit(1)

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    all_with_text = [c for c in chunks if (c.get("text") or "").strip()]
    expected_total = len(all_with_text)
    valid_ids = {c["chunk_id"] for c in all_with_text}
    unique_expected = len(valid_ids)
    if unique_expected != expected_total:
        _log(
            f"Note: {expected_total - unique_expected} duplicate chunk_id rows "
            f"({expected_total} rows, {unique_expected} unique ids)"
        )

    embeddings: dict[str, list[float]] = {}
    metadata: dict[str, dict] = {}
    if EMBEDDINGS_FILE.exists():
        try:
            existing = json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))
            raw_emb = dict(existing.get("embeddings", {}))
            raw_meta = dict(existing.get("metadata", {}))
            stale = set(raw_emb) - valid_ids
            if stale:
                _log(f"Pruning {len(stale)} stale vectors not in current chunk corpus")
            embeddings = {k: v for k, v in raw_emb.items() if k in valid_ids}
            metadata = {k: v for k, v in raw_meta.items() if k in valid_ids}
            _log(
                f"Resuming from {EMBEDDINGS_FILE}: {len(embeddings)} vectors "
                f"for current corpus ({len(raw_emb)} on disk before prune)"
            )
        except (json.JSONDecodeError, OSError) as exc:
            _log(f"WARNING: could not load existing embeddings ({exc}); starting fresh")

    to_embed = [c for c in all_with_text if c["chunk_id"] not in embeddings]
    _log(
        f"Embedding {len(to_embed)} new chunks "
        f"({len(embeddings)}/{expected_total} already done)"
    )

    if not to_embed:
        missing = valid_ids - set(embeddings)
        if missing:
            _log(f"ERROR: {len(missing)} current chunks still lack embeddings")
            sys.exit(1)
        _log(f"All {expected_total} chunks already embedded — saving pruned corpus")
        dim = EMBEDDING_DIMENSION
        if embeddings:
            dim = len(next(iter(embeddings.values())))
        out = {
            "model": EMBEDDING_MODEL,
            "dimension": dim,
            "total_vectors": len(embeddings),
            "embeddings": embeddings,
            "metadata": metadata,
        }
        EMBEDDINGS_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        _log(f"Pruned checkpoint saved: {len(embeddings)} vectors -> {EMBEDDINGS_FILE}")
        return out

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
        trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE,
    )

    save_every = int(os.getenv("EMBED_SAVE_EVERY", "200"))
    t0 = time.perf_counter()
    vectors_at_last_save = len(embeddings)

    for start in range(0, len(to_embed), EMBEDDING_BATCH):
        batch_idx = start // EMBEDDING_BATCH
        batch_chunks = to_embed[start : start + EMBEDDING_BATCH]
        batch_texts = [
            _embedding_prefixes()[1] + build_chunk_embedding_text(c) for c in batch_chunks
        ]
        batch_ids = [c["chunk_id"] for c in batch_chunks]
        try:
            vectors = model.encode(
                batch_texts,
                batch_size=min(EMBEDDING_BATCH, len(batch_texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            _save_checkpoint(
                embeddings,
                metadata,
                model,
                EMBEDDINGS_FILE,
                reason=f"encode failure at batch {batch_idx} (offset {start})",
            )
            _log(
                f"ERROR: model.encode() failed at batch_index={batch_idx}, "
                f"chunk_offset={start}, batch_size={len(batch_chunks)}: {exc}"
            )
            raise

        for cid, vec, chunk in zip(batch_ids, vectors, batch_chunks):
            embeddings[cid] = vec.tolist()
            metadata[cid] = {
                "type": chunk.get("chunk_type"),
                "regulation": chunk.get("regulation"),
                "heading_path": chunk.get("heading_path"),
                "parent_id": chunk.get("parent_id"),
            }

        done = len(embeddings)
        corpus_done = len(valid_ids & set(embeddings))
        if corpus_done % 50 == 0 or corpus_done == expected_total:
            _log(f"Progress: {corpus_done}/{expected_total} vectors embedded")

        if save_every and (done - vectors_at_last_save) >= save_every:
            _save_checkpoint(embeddings, metadata, model, EMBEDDINGS_FILE, reason="periodic")
            vectors_at_last_save = done

        gc.collect()

    _save_checkpoint(embeddings, metadata, model, EMBEDDINGS_FILE, reason="final")
    elapsed = round(time.perf_counter() - t0, 1)
    _log(f"Embedding complete: {len(embeddings)} vectors in {elapsed}s")

    missing = valid_ids - set(embeddings)
    if missing:
        _log(
            f"ERROR: {len(missing)} chunks with non-empty text lack embeddings "
            f"(have {len(embeddings)}, expected {expected_total})"
        )
        sys.exit(1)
    if len(embeddings) != unique_expected:
        _log(
            f"ERROR: vector count mismatch — {len(embeddings)} vectors saved, "
            f"expected {unique_expected} unique chunk_ids ({expected_total} chunk rows)"
        )
        sys.exit(1)

    return {
        "model": EMBEDDING_MODEL,
        "dimension": _embedding_dimension(model),
        "total_vectors": len(embeddings),
        "embeddings": embeddings,
        "metadata": metadata,
    }


def run_incremental(new_chunks: list[dict], model=None) -> dict:
    """Embed only chunks not already present in EMBEDDINGS_FILE; merge and save."""
    existing: dict = {"embeddings": {}, "metadata": {}}
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, encoding="utf-8") as f:
            existing = json.load(f)

    known = set(existing.get("embeddings", {}))
    to_embed = [
        c for c in new_chunks
        if (c.get("text") or "").strip() and c.get("chunk_id") not in known
    ]
    if not to_embed:
        p(f"No new chunks to embed ({len(known)} vectors already on disk)")
        return existing

    from sentence_transformers import SentenceTransformer

    if model is None:
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu",
            trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE,
        )

    p(f"Incremental embed: {len(to_embed)} new chunks ({len(known)} existing)")
    embeddings = dict(existing.get("embeddings", {}))
    metadata = dict(existing.get("metadata", {}))
    t0 = time.perf_counter()

    for start in range(0, len(to_embed), EMBEDDING_BATCH):
        batch_chunks = to_embed[start : start + EMBEDDING_BATCH]
        batch_texts = [
            _embedding_prefixes()[1] + build_chunk_embedding_text(c) for c in batch_chunks
        ]
        batch_ids = [c["chunk_id"] for c in batch_chunks]
        vectors = model.encode(
            batch_texts,
            batch_size=min(EMBEDDING_BATCH, len(batch_texts)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for cid, vec, chunk in zip(batch_ids, vectors, batch_chunks):
            embeddings[cid] = vec.tolist()
            metadata[cid] = {
                "type": chunk.get("chunk_type"),
                "regulation": chunk.get("regulation"),
                "heading_path": chunk.get("heading_path"),
                "parent_id": chunk.get("parent_id"),
            }
        gc.collect()

    out = {
        "model": EMBEDDING_MODEL,
        "dimension": getattr(
            model, "get_embedding_dimension", model.get_sentence_embedding_dimension
        )(),
        "total_vectors": len(embeddings),
        "embeddings": embeddings,
        "metadata": metadata,
    }
    EMBEDDINGS_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    elapsed = round(time.perf_counter() - t0, 1)
    p(f"Merged {len(to_embed)} new vectors -> {EMBEDDINGS_FILE} ({elapsed}s, total={len(embeddings)})")
    return out


if __name__ == "__main__":
    run()
