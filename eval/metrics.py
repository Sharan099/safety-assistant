"""Retrieval ranking metrics: recall@k, precision@k, MRR, NDCG."""

from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = {x for x in gold if x}
    if not g:
        return 0.0
    hit = {x for x in retrieved[:k] if x in g}
    return len(hit) / len(g)


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    r = list(retrieved[:k])
    if not r:
        return 0.0
    g = {x for x in gold if x}
    return sum(1 for x in r if x in g) / len(r)


def mrr(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    g = {x for x in gold if x}
    if not g:
        return 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in g:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += (2**rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = {x for x in gold if x}
    if not g:
        return 0.0
    rels = [1.0 if x in g else 0.0 for x in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    # Ideal DCG for binary: |g ∩ retrieved[:k]| ones at the front, else min(k,|g|) ones.
    n_ideal = min(k, len(g))
    ideal_rels = [1.0] * n_ideal + [0.0] * max(0, k - n_ideal)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg <= 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
