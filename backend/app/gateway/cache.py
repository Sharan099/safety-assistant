"""
Semantic cache for the gateway.

Design constraints (from the architecture):
  - REUSE the existing BGE embedding model (BAAI/bge-base-en-v1.5). No second
    embedding model is introduced; the embed function is injected.
  - Use Redis. Primary path is a RediSearch (redis-stack) cosine vector index;
    if RediSearch is unavailable we fall back to a bounded NumPy cosine scan over
    recent entries so the cache still works on plain Redis.
  - Cache stores prompt, answer, citations, grounding metadata, model.
  - Lookups are namespaced by the detected regulation scope so a UN R14 answer is
    never served for a UN R16 query.

The cache fails open: any Redis/RediSearch error degrades to "miss" and never
breaks a chat request.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

import numpy as np
from loguru import logger

from backend.app.gateway import config as cfg

EmbedFn = Callable[[str], "np.ndarray | list[float]"]


def _scope_tag(scope: list[str] | None) -> str:
    if not scope:
        return "global"
    return "+".join(sorted(scope))


def _normalise_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _to_unit_vector(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return arr
    return arr / norm


class SemanticCache:
    def __init__(self, embed_fn: EmbedFn, redis_client: Any | None = None) -> None:
        self._embed_fn = embed_fn
        self._redis = redis_client
        self._use_search = False
        self._ready = False
        self._dim = cfg.CACHE_EMBED_DIM

    # ───────────────────────── lifecycle ─────────────────────────
    def connect(self) -> None:
        if self._ready:
            return
        if self._redis is None:
            try:
                import redis

                self._redis = redis.Redis.from_url(
                    cfg.REDIS_URL, decode_responses=False
                )
                self._redis.ping()
            except Exception as exc:
                logger.warning(f"Semantic cache disabled (no Redis): {exc}")
                self._redis = None
                return
        self._ensure_index()
        self._ready = True

    def available(self) -> bool:
        return self._redis is not None

    def _ensure_index(self) -> None:
        """Create the RediSearch vector index if redis-stack is present."""
        try:
            from redis.commands.search.field import (
                NumericField,
                TagField,
                TextField,
                VectorField,
            )
            from redis.commands.search.indexDefinition import (
                IndexDefinition,
                IndexType,
            )

            try:
                self._redis.ft(cfg.CACHE_INDEX_NAME).info()
                self._use_search = True
                return
            except Exception:
                pass  # index does not exist yet -> create it

            schema = (
                TextField("prompt"),
                TagField("scope"),
                NumericField("created_at"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            definition = IndexDefinition(
                prefix=[cfg.CACHE_KEY_PREFIX], index_type=IndexType.HASH
            )
            self._redis.ft(cfg.CACHE_INDEX_NAME).create_index(
                schema, definition=definition
            )
            self._use_search = True
            logger.info("Gateway semantic cache: RediSearch vector index ready")
        except Exception as exc:
            # Plain Redis (no RediSearch module): use the NumPy scan fallback.
            self._use_search = False
            logger.info(f"Gateway semantic cache: scan fallback mode ({exc})")

    # ───────────────────────── public API ─────────────────────────
    def lookup(
        self, prompt: str, scope: list[str] | None = None
    ) -> dict[str, Any] | None:
        if not cfg.ENABLE_SEMANTIC_CACHE:
            return None
        self.connect()
        if self._redis is None:
            return None
        try:
            qvec = _to_unit_vector(self._embed_fn(_normalise_prompt(prompt)))
            if self._use_search:
                hit = self._search_lookup(qvec, scope)
            else:
                hit = self._scan_lookup(qvec, scope)
            return hit
        except Exception as exc:
            logger.warning(f"Cache lookup failed (treating as miss): {exc}")
            return None

    def store(
        self,
        *,
        prompt: str,
        answer: str,
        model: str,
        scope: list[str] | None = None,
        citations: list[dict] | None = None,
        grounding: dict | None = None,
    ) -> None:
        if not cfg.ENABLE_SEMANTIC_CACHE:
            return
        self.connect()
        if self._redis is None or not answer:
            return
        try:
            vec = _to_unit_vector(self._embed_fn(_normalise_prompt(prompt)))
            key = self._key(prompt, scope)
            payload = {
                "prompt": prompt.encode("utf-8"),
                "answer": answer.encode("utf-8"),
                "model": model.encode("utf-8"),
                "scope": _scope_tag(scope).encode("utf-8"),
                "citations": json.dumps(citations or []).encode("utf-8"),
                "grounding": json.dumps(grounding or {}).encode("utf-8"),
                "created_at": str(int(time.time())).encode("utf-8"),
                "embedding": vec.astype(np.float32).tobytes(),
            }
            self._redis.hset(key, mapping=payload)
            if cfg.CACHE_TTL_S > 0:
                self._redis.expire(key, cfg.CACHE_TTL_S)
        except Exception as exc:
            logger.warning(f"Cache store failed (non-fatal): {exc}")

    # ───────────────────────── internals ─────────────────────────
    def _key(self, prompt: str, scope: list[str] | None) -> str:
        h = hashlib.sha256(
            f"{_scope_tag(scope)}::{_normalise_prompt(prompt)}".encode("utf-8")
        ).hexdigest()
        return f"{cfg.CACHE_KEY_PREFIX}{h}"

    def _search_lookup(
        self, qvec: np.ndarray, scope: list[str] | None
    ) -> dict[str, Any] | None:
        from redis.commands.search.query import Query

        tag = _scope_tag(scope)
        base = f"(@scope:{{{_escape_tag(tag)}}})" if tag != "global" else "*"
        q = (
            Query(f"{base}=>[KNN 1 @embedding $vec AS dist]")
            .sort_by("dist")
            .return_fields("dist", "answer", "citations", "grounding", "model",
                           "prompt")
            .dialect(2)
        )
        res = self._redis.ft(cfg.CACHE_INDEX_NAME).search(
            q, query_params={"vec": qvec.astype(np.float32).tobytes()}
        )
        if not res.docs:
            return None
        doc = res.docs[0]
        # RediSearch COSINE distance = 1 - cosine_similarity.
        similarity = 1.0 - float(getattr(doc, "dist", 1.0))
        if similarity < cfg.CACHE_SIM_THRESHOLD:
            return None
        return _decode_hit(doc, similarity)

    def _scan_lookup(
        self, qvec: np.ndarray, scope: list[str] | None
    ) -> dict[str, Any] | None:
        tag = _scope_tag(scope).encode("utf-8")
        best: tuple[float, dict] | None = None
        scanned = 0
        cursor = 0
        pattern = f"{cfg.CACHE_KEY_PREFIX}*".encode("utf-8")
        while True:
            cursor, keys = self._redis.scan(
                cursor=cursor, match=pattern, count=128
            )
            for key in keys:
                if scanned >= cfg.CACHE_FALLBACK_SCAN_LIMIT:
                    break
                entry = self._redis.hgetall(key)
                if not entry:
                    continue
                if entry.get(b"scope") not in (tag, b"global"):
                    continue
                emb = entry.get(b"embedding")
                if not emb:
                    continue
                vec = np.frombuffer(emb, dtype=np.float32)
                if vec.shape[0] != qvec.shape[0]:
                    continue
                sim = float(np.dot(qvec, _to_unit_vector(vec)))
                scanned += 1
                if best is None or sim > best[0]:
                    best = (sim, _decode_hit_dict(entry, sim))
            if cursor == 0 or scanned >= cfg.CACHE_FALLBACK_SCAN_LIMIT:
                break
        if best and best[0] >= cfg.CACHE_SIM_THRESHOLD:
            return best[1]
        return None


def _escape_tag(tag: str) -> str:
    # RediSearch tag values: escape special chars.
    for ch in "+-.@{}[]() ":
        tag = tag.replace(ch, f"\\{ch}")
    return tag


def _decode_hit(doc, similarity: float) -> dict[str, Any]:
    def _g(name: str, default: str = "") -> str:
        val = getattr(doc, name, default)
        return val.decode("utf-8") if isinstance(val, bytes) else (val or default)

    return {
        "answer": _g("answer"),
        "model": _g("model"),
        "prompt": _g("prompt"),
        "citations": json.loads(_g("citations", "[]") or "[]"),
        "grounding": json.loads(_g("grounding", "{}") or "{}"),
        "similarity": round(similarity, 4),
    }


def _decode_hit_dict(entry: dict[bytes, bytes], similarity: float) -> dict[str, Any]:
    def _g(name: str, default: str = "") -> str:
        v = entry.get(name.encode("utf-8"))
        return v.decode("utf-8") if isinstance(v, bytes) else default

    return {
        "answer": _g("answer"),
        "model": _g("model"),
        "prompt": _g("prompt"),
        "citations": json.loads(_g("citations", "[]") or "[]"),
        "grounding": json.loads(_g("grounding", "{}") or "{}"),
        "similarity": round(similarity, 4),
    }
