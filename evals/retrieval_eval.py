"""Retrieval evaluation — TRD.md Section 14/Phase 12: Recall@5, Recall@10, MRR.

Distinct from tests/retrieval (which asserts a couple of known-good
queries): this runs a golden query set end to end and reports aggregate
metrics, the way IMPLEMENTATION_PLAN.md Phase 12 describes ("Create a
golden dataset... For every answer verify: did it retrieve the correct
document?").

Usage:
    uv run python evals/retrieval_eval.py
"""

from __future__ import annotations

import pathlib
import sys

import yaml
from sqlalchemy.orm import Session

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.domain.db import get_engine  # noqa: E402
from packages.retrieval.search import retrieve  # noqa: E402

GOLDEN_SET_PATH = ROOT / "evals" / "golden_retrieval_set.yaml"


def _reciprocal_rank(document_keys: list[str], expected: str) -> float:
    for rank, key in enumerate(document_keys, start=1):
        if key == expected:
            return 1.0 / rank
    return 0.0


def main() -> None:
    with GOLDEN_SET_PATH.open("r", encoding="utf-8") as f:
        golden = yaml.safe_load(f)["cases"]

    with Session(get_engine()) as session:
        recall_5_hits = 0
        recall_10_hits = 0
        reciprocal_ranks: list[float] = []

        print(f"{'query':<55} {'expected':<20} {'top_hit':<20} {'rank':>5}")
        print("-" * 105)

        for case in golden:
            results = retrieve(session, case["query"], limit=10)
            document_keys = [r.document_key for r in results]
            expected = case["expected_document_key"]

            rank = next((i + 1 for i, k in enumerate(document_keys) if k == expected), None)
            recall_5_hits += 1 if rank is not None and rank <= 5 else 0
            recall_10_hits += 1 if rank is not None and rank <= 10 else 0
            reciprocal_ranks.append(_reciprocal_rank(document_keys, expected))

            top_hit = document_keys[0] if document_keys else "(none)"
            print(f"{case['query']:<55} {expected:<20} {top_hit:<20} {rank if rank else '-':>5}")

        n = len(golden)
        recall_5 = recall_5_hits / n
        recall_10 = recall_10_hits / n
        mrr = sum(reciprocal_ranks) / n

        print("-" * 105)
        print(f"Recall@5:  {recall_5:.2f} ({recall_5_hits}/{n})")
        print(f"Recall@10: {recall_10:.2f} ({recall_10_hits}/{n})")
        print(f"MRR:       {mrr:.3f}")
        print()
        print(
            "Note: retrieval uses the interim HashingEmbeddingProvider (docs/ADR/0007) — "
            "word-overlap only, no semantics. Low scores here motivate benchmarking a real "
            "embedding model (TRD.md Section 20), not a retrieval-code bug."
        )


if __name__ == "__main__":
    main()
