"""Scorecard helpers for retrieval-only reports (`eval.run_retrieval_only`).

Not the CI / production gate — that is ``eval.run_full`` → ``results.json`` +
``eval/thresholds.yaml``. ``eval.quality_gate`` still reads this shape for a
legacy fixture smoke in ``main.yml`` only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def build_scorecard(
    *,
    tag: str,
    retrieval: dict[str, Any],
    ragas: dict[str, Any] | None = None,
    deepeval: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created = datetime.now(timezone.utc)
    retrieval_avg = dict(
        retrieval.get("averages_chunk_gold") or retrieval.get("averages") or {}
    )
    excluded_avg = dict(retrieval.get("averages_excluded_non_retrieval") or {})
    # Canonical retrieval keys = chunk-gold categories only (never blended).
    scorecard = {
        "recall@5": retrieval_avg.get("recall@5"),
        "precision@5": retrieval_avg.get("precision@5"),
        "mrr": retrieval_avg.get("mrr"),
        "ndcg@10": retrieval_avg.get("ndcg@10"),
        "retrieval_quality_scope": "chunk-gold categories only",
        "chunk_gold_n_cases": retrieval_avg.get("n_cases"),
        "excluded_non_retrieval_n_cases": excluded_avg.get("n_cases"),
    }
    if ragas and ragas.get("averages"):
        for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            scorecard[f"ragas_{k}"] = ragas["averages"].get(k)
    if deepeval and deepeval.get("averages"):
        for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            scorecard[f"deepeval_{k}"] = deepeval["averages"].get(k)

    return {
        "created_at": created.isoformat(),
        "tag": tag,
        "mode": "full" if (ragas or deepeval) else "retrieval-only",
        "scorecard": scorecard,
        "retrieval": retrieval,
        "ragas": ragas,
        "deepeval": deepeval,
        "meta": meta or {},
    }


def print_scorecard(report: dict[str, Any]) -> None:
    sc = report.get("scorecard") or {}
    retrieval = report.get("retrieval") or {}
    print()
    print("=" * 64)
    print(f"SCORECARD  tag={report.get('tag')}  mode={report.get('mode')}")
    print(f"created_at={report.get('created_at')}")
    print("=" * 64)
    cg = retrieval.get("averages_chunk_gold") or retrieval.get("averages") or {}
    ex = retrieval.get("averages_excluded_non_retrieval") or {}
    print(
        "Retrieval quality (chunk-gold categories only)  "
        f"n={cg.get('n_cases')}"
    )
    print(
        f"  mrr={cg.get('mrr')}  recall@5={cg.get('recall@5')}  "
        f"precision@5={cg.get('precision@5')}  ndcg@10={cg.get('ndcg@10')}"
    )
    print(
        "Excluded from retrieval quality (non-retrieval categories)  "
        f"n={ex.get('n_cases')}  "
        "(numeric_safety, guardrail, prompt_injection, "
        "hallucination_probe, out_of_scope — not blended above)"
    )
    print("-" * 64)
    print(f"{'metric':<28} {'score':>10}")
    print("-" * 64)
    for key, val in sc.items():
        if val is None:
            cell = "—"
        else:
            try:
                cell = f"{float(val):.4f}"
            except (TypeError, ValueError):
                cell = str(val)
        print(f"{key:<28} {cell:>10}")
    print("=" * 64)


def save_scorecard(report: dict[str, Any], *, out: Path | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = (report.get("tag") or "run").replace(" ", "_")
    mode = report.get("mode") or "eval"
    path = out or (RESULTS_DIR / f"{tag}-{mode}-{ts}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also refresh a stable "latest" pointer for quick diffs.
    latest = RESULTS_DIR / "latest.json"
    latest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
