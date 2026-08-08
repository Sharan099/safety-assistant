"""Input/output validation and simple in-memory rate limiting."""

from __future__ import annotations

import re
import time
from collections import defaultdict

from fastapi import HTTPException

from app.config import settings

_INJECTION_QUICK = re.compile(
    r"(ignore\s+(all\s+)?instructions|jailbreak|reveal\s+(the\s+)?system\s+prompt)",
    re.I,
)


class RateLimiter:
    """Per-IP sliding window — good enough for single-instance deployments."""

    def __init__(self, limit_per_minute: int | None = None):
        self.limit = limit_per_minute or settings.RATE_LIMIT_PER_MINUTE
        self._hits: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str) -> None:
        now = time.time()
        window = self._hits[client_id]
        self._hits[client_id] = [t for t in window if now - t < 60]
        if len(self._hits[client_id]) >= self.limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
        self._hits[client_id].append(now)


rate_limiter = RateLimiter()


def validate_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="Query must not be empty.")
    if len(q) > settings.MAX_QUERY_CHARS:
        raise HTTPException(status_code=422, detail=f"Query exceeds {settings.MAX_QUERY_CHARS} characters.")
    if _INJECTION_QUICK.search(q):
        raise HTTPException(status_code=400, detail="Query rejected by input validation.")
    return q


def validate_answer(answer: str) -> str:
    if not answer:
        return "I could not generate an answer from the retrieved sources."
    if len(answer) > 50_000:
        return answer[:50_000]
    return answer
