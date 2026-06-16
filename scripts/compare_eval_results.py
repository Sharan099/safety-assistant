#!/usr/bin/env python3
"""Compare two evaluation JSON artifacts side-by-side."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(v: float | None, digits: int = 4) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def delta(new: float | None, old: float | None) -> str:
    if new is None or old is None:
        return "—"
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def main() -> None:
    baseline = ROOT / "output/evaluation/rag_eval_20_results.bge_baseline.json"
    upgraded = ROOT / "output/evaluation/rag_eval_20_results.nomic_v2m3.json"
    if len(sys.argv) >= 3:
        baseline = Path(sys.argv[1])
        upgraded = Path(sys.argv[2])

    b, n = load(baseline), load(upgraded)
    print("=" * 72)
    print("RAG EVAL COMPARISON")
    print("=" * 72)
    print(f"Baseline : {b.get('models')}  ({baseline.name})")
    print(f"Upgraded : {n.get('models')}  ({upgraded.name})")
    print(f"Mode     : baseline={b.get('evaluation_mode')}")
    print(f"           upgraded={n.get('evaluation_mode')}")
    print()

    print("RAGAS (primary)")
    print(f"{'Metric':<22} {'Baseline':>10} {'Upgraded':>10} {'Delta':>10}")
    print("-" * 54)
    for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        bv = b.get("ragas", {}).get(k)
        nv = n.get("ragas", {}).get(k)
        print(f"{k:<22} {fmt(bv):>10} {fmt(nv):>10} {delta(nv, bv):>10}")
    print(f"{'overall_score':<22} {fmt(b.get('overall_score')):>10} {fmt(n.get('overall_score')):>10} {delta(n.get('overall_score'), b.get('overall_score')):>10}")
    print()

    print("Ablation — retrieval proxies")
    for label, side in ("Baseline", b), ("Upgraded", n):
        ab = side.get("ablation", {})
        sem = ab.get("semantic_only", {})
        hyb = ab.get("hybrid_rrf_rerank", {})
        print(
            f"  {label}: semantic recall={fmt(sem.get('avg_context_recall_proxy'))} "
            f"hybrid recall={fmt(hyb.get('avg_context_recall_proxy'))} "
            f"lift={ab.get('context_recall_lift_pct')}% "
            f"hybrid latency={fmt(hyb.get('avg_latency_ms'), 0)}ms"
        )
    print()

    print("Latency (ms)")
    for k in ("semantic_avg_ms", "hybrid_avg_ms", "retrieval_p95_ms", "pipeline_p95_ms"):
        bv = b.get("latency", {}).get(k)
        nv = n.get("latency", {}).get(k)
        print(f"  {k:<22} baseline={fmt(bv, 0):>8}  upgraded={fmt(nv, 0):>8}  delta={delta(nv, bv):>10}")


if __name__ == "__main__":
    main()
