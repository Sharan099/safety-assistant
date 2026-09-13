"""A/B: baseline (`content`) vs summary-augmented (`sac_v1`) retrieval on identical queries and
gold labels — deterministic, no LLM at query time. Everything else in the pipeline is held equal.

uv run python scripts/eval/sac_ab.py --datasets evals/datasets/document_mismatch_v1.yaml \
    evals/datasets/regulatory_v2.yaml --legs full hybrid_rrf

Writes one retrieval report per (dataset, representation) under evals/results/ plus
evals/results/sac_ab_<stamp>.json with the comparison table and the per-case DRM diff.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

from sqlalchemy.orm import Session

from safety_assistant.config import get_settings
from safety_assistant.evaluation.dataset import load_dataset
from safety_assistant.evaluation.retrieval_eval import EvalReport, run_evaluation, write_report
from safety_assistant.persistence import get_engine
from safety_assistant.retrieval import RetrievalConfig

ROWS = [
    ("Document Recall@1", "doc_recall@1"),
    ("Document Recall@3", "doc_recall@3"),
    ("Document Recall@5", "doc_recall@5"),
    ("Document MRR", "doc_mrr"),
    ("DRM rate @1", "drm@1"),
    ("DRM rate @5", "drm@5"),
    ("Passage Recall@5", "recall@5"),
    ("Passage Recall@10", "recall@10"),
    ("Passage Recall@20", "recall@20"),
    ("Passage MRR", "mrr"),
    ("nDCG@10", "ndcg@10"),
    ("p50 latency ms", "latency_p50_ms"),
    ("p95 latency ms", "latency_p95_ms"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["evals/datasets/document_mismatch_v1.yaml"])
    ap.add_argument("--legs", nargs="+", default=["full", "hybrid_rrf"])
    ap.add_argument("--out", default="evals/results")
    args = ap.parse_args(argv)
    out_dir = pathlib.Path(args.out)
    base = RetrievalConfig.from_settings(get_settings())
    datasets: dict[str, object] = {}
    comparison: dict[str, object] = {"legs": args.legs, "datasets": datasets}
    with Session(get_engine()) as session:
        for ds_path in args.datasets:
            dataset = load_dataset(pathlib.Path(ds_path))
            reports: dict[str, EvalReport] = {}
            for rep in ("content", "sac_v1"):
                cfg = RetrievalConfig(**{**base.__dict__, "representation": rep})
                reports[rep] = run_evaluation(session, dataset, legs=args.legs, base_config=cfg)
                write_report(reports[rep], out_dir, name=f"retrieval_{rep}")
            datasets[dataset.dataset_version] = _compare(dataset.dataset_version, reports, args.legs)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"sac_ab_{stamp}.json"
    path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    print(f"written: {path}")
    return 0


def _compare(name: str, reports: dict[str, EvalReport], legs: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for leg in legs:
        a = next(lr for lr in reports["content"].legs if lr.leg == leg)
        b = next(lr for lr in reports["sac_v1"].legs if lr.leg == leg)
        n, elig = a.aggregate.get("n") or 0, a.aggregate.get("n_drm_eligible") or 0
        print(f"\n{name} — leg {leg}  (n={int(n)}, DRM-eligible {int(elig)})")
        print(f"{'Metric':<22}{'Baseline':>10}{'SAC':>10}{'Delta':>10}")
        table: dict[str, dict[str, float | None]] = {}
        for label, key in ROWS:
            va, vb = a.aggregate.get(key), b.aggregate.get(key)
            delta = None if va is None or vb is None else vb - va
            table[key] = {"baseline": va, "sac": vb, "delta": delta}
            print(f"{label:<22}{_f(va):>10}{_f(vb):>10}{_f(delta, sign=True):>10}")
        # per-case DRM diff: which queries flipped
        ca = {c.case_id: c for c in a.cases}
        fixed, broken = [], []
        for cb in b.cases:
            x, y = ca[cb.case_id].metrics.get("drm@1"), cb.metrics.get("drm@1")
            if x == 1.0 and y == 0.0:
                fixed.append(cb.case_id)
            elif x == 0.0 and y == 1.0:
                broken.append(cb.case_id)
        print(f"DRM@1 fixed by SAC: {len(fixed)} {fixed[:12]}")
        print(f"DRM@1 broken by SAC: {len(broken)} {broken[:12]}")
        out[leg] = {"table": table, "drm1_fixed": fixed, "drm1_broken": broken}
    return out


def _f(v: float | None, sign: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.3f}" if sign else f"{v:.3f}"


if __name__ == "__main__":
    sys.exit(main())
