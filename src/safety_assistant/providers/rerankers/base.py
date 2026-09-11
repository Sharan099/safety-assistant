from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RerankCandidate:
    id: uuid.UUID
    content: str
    authority_level: str
    fused_score: float
    normative: bool | None = None
    chunk_type: str = "TEXT"


class Reranker(Protocol):
    model_name: str
    model_version: str

    def score(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        """One score per candidate, same order. Higher is better."""
        ...
