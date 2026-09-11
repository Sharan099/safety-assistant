"""Apply a `Reranker` and record model identity + latency."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from safety_assistant.providers.rerankers import RerankCandidate, Reranker


@dataclass
class RerankObservation:
    model_name: str
    model_version: str
    latency_ms: float
    scores: dict[uuid.UUID, float]


def apply_reranker(
    reranker: Reranker | None, query: str, candidates: list[RerankCandidate]
) -> RerankObservation | None:
    if reranker is None or not candidates:
        return None
    t0 = time.perf_counter()
    raw = reranker.score(query, candidates)
    return RerankObservation(
        model_name=reranker.model_name,
        model_version=reranker.model_version,
        latency_ms=(time.perf_counter() - t0) * 1000,
        scores={c.id: s for c, s in zip(candidates, raw, strict=True)},
    )
