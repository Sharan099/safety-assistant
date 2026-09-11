"""Reciprocal Rank Fusion (k=60 default). Ranks only — never raw-score sums."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[uuid.UUID]], *, k: int = RRF_K, weights: Sequence[float] | None = None
) -> dict[uuid.UUID, float]:
    weights = list(weights) if weights else [1.0] * len(ranked_lists)
    scores: dict[uuid.UUID, float] = {}
    for ranked, w in zip(ranked_lists, weights, strict=True):
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return scores
