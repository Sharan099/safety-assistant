"""Exact-match (+ optional semantic) response cache backed by SQLite.

NOTE: Portkey simple cache on QUERY_REWRITE / FINAL_ANSWER configs is now the
primary LLM cache (see ``generation/llm_client.py`` + ``config/portkey/*.json``).
That gateway cache is namespaced by ``cache_version`` (bumped on every successful
ingest) so newly uploaded regulations never reuse a stale "not found" completion.

This module is an *optional local fallback* for the full retrieve+answer pipeline
when the Portkey gateway is unreachable. Leave ``ANSWER_CACHE=0`` (default in
``.env.example``) unless you explicitly want that second layer — two caches that
can disagree are worse than one authoritative gateway cache.

When enabled: cache key = sha256(cache_version | normalized question | regulation).
``bump_cache_version`` still changes the key space on ingest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "response_cache.sqlite3"

_lock = threading.Lock()
_db_path: Path | None = None

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def _path() -> Path:
    global _db_path
    if _db_path is None:
        _db_path = Path(os.getenv("RESPONSE_CACHE_DB") or DEFAULT_DB)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    return _db_path


def _enabled() -> bool:
    # Default off: Portkey gateway simple cache is primary (see module docstring).
    return os.getenv("ANSWER_CACHE", "0") not in {"0", "false", "False", ""}


def semantic_enabled() -> bool:
    if not _enabled():
        return False
    return os.getenv("SEMANTIC_CACHE", "1") not in {"0", "false", "False"}


def semantic_threshold() -> float:
    try:
        return float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
    except ValueError:
        return 0.95


def normalize_question(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def cache_key(question: str, *, regulation_id: str | None = None) -> str:
    from api.cache_version import get_cache_version

    ver = get_cache_version()
    raw = f"v{ver}|{normalize_question(question)}|{regulation_id or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    cache_version INTEGER NOT NULL,
                    question_norm TEXT NOT NULL,
                    question_raw TEXT NOT NULL,
                    regulation_id TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    embedding BLOB,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_responses_version "
                "ON responses(cache_version, regulation_id)"
            )
            conn.commit()
        finally:
            conn.close()


def _pack_embedding(vec: Sequence[float] | None) -> bytes | None:
    if not vec:
        return None
    return struct.pack(f"{len(vec)}f", *[float(x) for x in vec])


def _unpack_embedding(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    n = len(blob) // 4
    if n <= 0 or len(blob) != n * 4:
        return None
    return list(struct.unpack(f"{n}f", blob))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def get_exact(
    question: str,
    *,
    regulation_id: str | None = None,
) -> dict[str, Any] | None:
    """Exact-match lookup. Returns payload or None."""
    if not _enabled():
        return None
    init_db()
    key = cache_key(question, regulation_id=regulation_id)
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM responses WHERE cache_key = ?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return None
    try:
        data = json.loads(row["payload_json"])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    data["cache_hit"] = "exact"
    logger.info(
        "response cache exact hit key=%s question=%r regulation_id=%r",
        key[:12],
        normalize_question(question)[:80],
        regulation_id,
    )
    return data


def put_exact(
    question: str,
    payload: dict[str, Any],
    *,
    regulation_id: str | None = None,
    embedding: Sequence[float] | None = None,
) -> str:
    """Store full response payload. Returns cache key."""
    if not _enabled():
        return ""
    init_db()
    from api.cache_version import get_cache_version

    ver = get_cache_version()
    key = cache_key(question, regulation_id=regulation_id)
    q_norm = normalize_question(question)
    data = {
        **payload,
        "cached_at": time.time(),
        "question": question,
        "cache_version": ver,
        "regulation_id": regulation_id or "",
    }
    blob = _pack_embedding(embedding)
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO responses
                (cache_key, cache_version, question_norm, question_raw,
                 regulation_id, payload_json, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    ver,
                    q_norm,
                    question,
                    regulation_id or "",
                    json.dumps(data, ensure_ascii=False),
                    blob,
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return key


def get_semantic(
    question: str,
    *,
    regulation_id: str | None = None,
    embed_fn: EmbedFn | None = None,
    threshold: float | None = None,
) -> dict[str, Any] | None:
    """Nearest cached question by cosine similarity within current cache_version."""
    if not semantic_enabled():
        return None
    init_db()
    from api.cache_version import get_cache_version

    ver = get_cache_version()
    thr = semantic_threshold() if threshold is None else threshold
    q_norm = normalize_question(question)
    if not q_norm:
        return None

    if embed_fn is None:
        from ingestion.embed_upsert import Embedder

        embed_fn = Embedder().embed

    q_vecs = embed_fn([q_norm])
    if not q_vecs:
        return None
    q_emb = q_vecs[0]

    reg = regulation_id or ""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT question_raw, question_norm, payload_json, embedding
                FROM responses
                WHERE cache_version = ? AND regulation_id = ?
                  AND embedding IS NOT NULL
                """,
                (ver, reg),
            ).fetchall()
        finally:
            conn.close()

    best_sim = -1.0
    best_row: sqlite3.Row | None = None
    for row in rows:
        emb = _unpack_embedding(row["embedding"])
        if not emb:
            continue
        # Skip exact normalized duplicate — exact path should have caught it.
        if row["question_norm"] == q_norm:
            continue
        sim = cosine_similarity(q_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_row = row

    if best_row is None or best_sim < thr:
        return None

    try:
        data = json.loads(best_row["payload_json"])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data.get("answer"):
        return None

    cached_q = str(best_row["question_raw"] or best_row["question_norm"])
    logger.info(
        "semantic cache hit query=%r cached_question=%r similarity=%.4f "
        "threshold=%.4f regulation_id=%r cache_version=%s",
        question,
        cached_q,
        best_sim,
        thr,
        regulation_id,
        ver,
    )
    data = {
        **data,
        "cache_hit": "semantic",
        "semantic_similarity": round(best_sim, 6),
        "semantic_cached_question": cached_q,
    }
    return data


def lookup(
    question: str,
    *,
    regulation_id: str | None = None,
    embed_fn: EmbedFn | None = None,
    allow_semantic: bool = True,
) -> dict[str, Any] | None:
    """Exact first, then optional semantic."""
    hit = get_exact(question, regulation_id=regulation_id)
    if hit:
        return hit
    if allow_semantic:
        return get_semantic(
            question, regulation_id=regulation_id, embed_fn=embed_fn
        )
    return None


def store(
    question: str,
    payload: dict[str, Any],
    *,
    regulation_id: str | None = None,
    embed_fn: EmbedFn | None = None,
) -> str:
    """Write response; embed when semantic cache is enabled."""
    embedding: list[float] | None = None
    if semantic_enabled():
        try:
            if embed_fn is None:
                from ingestion.embed_upsert import Embedder

                embed_fn = Embedder().embed
            vecs = embed_fn([normalize_question(question)])
            if vecs:
                embedding = vecs[0]
        except Exception:  # noqa: BLE001
            logger.warning("response cache embedding failed; storing without vector")
            embedding = None
    return put_exact(
        question, payload, regulation_id=regulation_id, embedding=embedding
    )


def count_entries() -> int:
    """Number of rows in the response cache DB (0 if DB missing)."""
    path = _path()
    if not path.is_file():
        return 0
    init_db()
    with _lock:
        conn = _connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS n FROM responses").fetchone()
            return int(row["n"] if row else 0)
        finally:
            conn.close()


def clear_all() -> dict[str, Any]:
    """Delete all cached responses (or remove the DB file). Reports empty=True."""
    path = _path()
    removed = 0
    with _lock:
        if path.is_file():
            try:
                init_db()
                conn = _connect()
                try:
                    row = conn.execute("SELECT COUNT(*) AS n FROM responses").fetchone()
                    removed = int(row["n"] if row else 0)
                    conn.execute("DELETE FROM responses")
                    conn.commit()
                finally:
                    conn.close()
            except Exception:  # noqa: BLE001
                path.unlink(missing_ok=True)
                removed = -1
    remaining = count_entries()
    return {
        "path": str(path),
        "removed": removed,
        "remaining": remaining,
        "empty": remaining == 0,
    }


# Backward-compatible aliases used by generation.answer_cache
get_cached_answer = get_exact
put_cached_answer = put_exact
