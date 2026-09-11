"""Pure IR metrics over ranked id lists. Relevance is a set of ids."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence


def recall_at_k[T: Hashable](ranked: Sequence[T], relevant: set[T], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k[T: Hashable](ranked: Sequence[T], relevant: set[T], k: int) -> float | None:
    if not relevant:
        return None
    top = ranked[:k]
    return len(set(top) & relevant) / k if top else 0.0


def hit_at_k[T: Hashable](ranked: Sequence[T], relevant: set[T], k: int) -> float | None:
    if not relevant:
        return None
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank[T: Hashable](ranked: Sequence[T], relevant: set[T]) -> float | None:
    if not relevant:
        return None
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k[T: Hashable](ranked: Sequence[T], relevant: set[T], k: int) -> float | None:
    """Binary-gain nDCG."""
    if not relevant:
        return None
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def first_rank[T: Hashable](ranked: Sequence[T], relevant: set[T]) -> int | None:
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return i
    return None


def mean(values: list[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    return sum(real) / len(real) if real else None
