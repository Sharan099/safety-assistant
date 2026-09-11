"""Per-principal token bucket for query routes. In-process — one bucket per
worker, good enough to stop a single client from monopolising CPU-bound
retrieval. A shared limiter (Redis) is the upgrade path if replicas > 1 must
share a budget precisely (ponytail: in-process limiter, per-replica budget)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, status

from safety_assistant.api.dependencies.auth import Principal, get_principal
from safety_assistant.config import Settings, get_settings


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    per_minute: int
    _buckets: dict[str, _Bucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = _Bucket(tokens=float(self.per_minute), updated=now)
            b.tokens = min(float(self.per_minute), b.tokens + (now - b.updated) * self.per_minute / 60.0)
            b.updated = now
            if b.tokens < 1.0:
                return False
            b.tokens -= 1.0
            return True


_limiters: dict[int, RateLimiter] = {}


def _limiter(settings: Settings) -> RateLimiter:
    lim = _limiters.get(settings.rate_limit_per_minute)
    if lim is None:
        lim = _limiters[settings.rate_limit_per_minute] = RateLimiter(settings.rate_limit_per_minute)
    return lim


def rate_limited(
    principal: Principal = Depends(get_principal), settings: Settings = Depends(get_settings)
) -> Principal:
    if settings.rate_limit_per_minute > 0 and not _limiter(settings).allow(principal.subject):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded", headers={"Retry-After": "1"})
    return principal
