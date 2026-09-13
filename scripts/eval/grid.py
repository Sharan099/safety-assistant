"""Retrieval config grid on a gold set — deterministic, no LLM. Prints MRR/R@10/nDCG per config.

uv run python scripts/eval/grid.py --dataset evals/datasets/regulatory_v2.yaml --param sparse_weight 1.0 1.5 2.0 3.0
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
from typing import Any

from sqlalchemy.orm import Session

from safety_assistant.evaluation.dataset import load_dataset
from safety_assistant.evaluation.retrieval_eval import evaluate_leg
from safety_assistant.persistence import get_engine
from safety_assistant.retrieval import RetrievalConfig, RetrievalService
from safety_assistant.retrieval.sparse import get_index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="evals/datasets/regulatory_v2.yaml")
    ap.add_argument("--param", nargs="+", action="append", required=True, help="name v1 v2 ... (repeatable)")
    ap.add_argument("--fixed", nargs="*", default=[], help="name=value overrides applied to every run")
    args = ap.parse_args(argv)
    dataset = load_dataset(pathlib.Path(args.dataset))
    fixed: dict[str, Any] = {k: _coerce(v) for k, v in (f.split("=", 1) for f in args.fixed)}
    grid = [(p[0], [_coerce(v) for v in p[1:]]) for p in args.param]
    combos: list[dict[str, Any]] = [{}]
    for name, values in grid:
        combos = [{**c, name: v} for c in combos for v in values]
    with Session(get_engine()) as session:
        bm25 = get_index(session)
        print(f"{'config':<48}{'MRR':>8}{'R@5':>8}{'R@10':>8}{'nDCG10':>8}{'p50ms':>8}{'p95ms':>8}{'rerank%':>9}")
        for combo in combos:
            overrides: dict[str, Any] = {**fixed, **combo}
            cfg = dataclasses.replace(RetrievalConfig(), **overrides)
            rep = evaluate_leg(session, dataset, "grid", service=RetrievalService(config=cfg, bm25_index=bm25))
            a = rep.aggregate
            label = " ".join(f"{k}={v}" for k, v in {**fixed, **combo}.items()) or "default"
            lats = sorted(c.latency_ms for c in rep.cases)
            p50, p95 = lats[len(lats) // 2], lats[int(len(lats) * 0.95) - 1]
            rr = 100 * sum(1 for c in rep.cases if c.reranked) / len(rep.cases)
            print(
                f"{label:<48}{a['mrr']:>8.3f}{a['recall@5']:>8.3f}{a['recall@10']:>8.3f}{a['ndcg@10']:>8.3f}"
                f"{p50:>8.0f}{p95:>8.0f}{rr:>8.0f}%"
            )
    return 0


def _coerce(v: str) -> object:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


if __name__ == "__main__":
    sys.exit(main())
