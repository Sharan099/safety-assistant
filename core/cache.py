"""Simple in-process LRU cache for retrieval results."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

from app.config import settings


class RetrievalCache:
    def __init__(self, max_size: int | None = None):
        self.max_size = max_size or settings.RETRIEVAL_CACHE_SIZE
        self._store: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def _key(self, query: str, top_k: int, filters: dict[str, Any]) -> str:
        payload = json.dumps({"q": query, "k": top_k, "f": filters}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, query: str, top_k: int, filters: dict[str, Any]) -> list[dict[str, Any]] | None:
        if not settings.ENABLE_RETRIEVAL_CACHE:
            return None
        key = self._key(query, top_k, filters)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        return None

    def put(self, query: str, top_k: int, filters: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        if not settings.ENABLE_RETRIEVAL_CACHE:
            return
        key = self._key(query, top_k, filters)
        with self._lock:
            self._store[key] = chunks
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


retrieval_cache = RetrievalCache()
