"""Tune and A/B ``RERANK_MIN_SCORE`` against golden retrieval-only metrics.

Collects hybrid → rerank windows **once** (threshold applied offline), using the
same score floor site as production (``retrieval.rerank.rerank``).

Metrics use chunk match-key hit semantics (chunk_id / section_id / section_number)
so cases whose gold is ``expected_chunk_ids`` are scored correctly.

Usage::

    python scripts/ab_rerank_min_score.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from eval.gold import chunk_match_keys, gold_keys, load_golden_set
from eval.metrics import mean, mrr, ndcg_at_k
from retrieval.rerank import rerank
from retrieval.retrieve import hybrid_search

logger = logging.getLogger("ab_rerank")


def _precision_at_k_hits(flags: list[bool], k: int) -> float:
    r = flags[:k]
    if not r:
        return 0.0
    return sum(1 for x in r if x) / len(r)


def _recall_at_k_keys(kept_keys: list[set[str]], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    found: set[str] = set()
    for keys in kept_keys[:k]:
        found |= keys & gold
    return len(found) / len(gold)


def _mrr_hits(flags: list[bool]) -> float:
    for i, hit in enumerate(flags, start=1):
        if hit:
            return 1.0 / i
    return 0.0


def collect_scored_windows(*, top_n: int = 5) -> list[dict[str, Any]]:
    """hybrid_search → rerank(top_n) with min_score disabled."""
    prev = os.environ.pop("RERANK_MIN_SCORE", None)
    rows: list[dict[str, Any]] = []
    try:
        for case in load_golden_set():
            gold = gold_keys(case)
            if not gold:
                # Skip abstention / no-gold cases for retrieval A/B.
                continue
            hybrid = hybrid_search(
                case["question"],
                top_k=30,
                regulation_id=case.get("regulation_id"),
            )
            scored = rerank(
                case["question"],
                hybrid,
                top_n=top_n,
                min_score=None,
            )
            window = []
            for c in scored:
                keys = chunk_match_keys(c)
                window.append(
                    {
                        "chunk_id": c.chunk_id,
                        "score": float(c.score or 0.0),
                        "match_keys": sorted(keys),
                        "is_gold": bool(keys & gold),
                    }
                )
            rows.append(
                {
                    "id": case["id"],
                    "gold": sorted(gold),
                    "window": window,
                }
            )
            logger.info(
                "collected %s n=%d gold_in_window=%d",
                case["id"],
                len(window),
                sum(1 for w in window if w["is_gold"]),
            )
    finally:
        if prev is not None:
            os.environ["RERANK_MIN_SCORE"] = prev
        else:
            os.environ.pop("RERANK_MIN_SCORE", None)
    return rows


def apply_cutoff(
    rows: list[dict[str, Any]],
    min_score: float | None,
    *,
    score_margin: float | None = None,
) -> dict[str, Any]:
    per_case: list[dict[str, float]] = []
    for row in rows:
        gold = set(row["gold"])
        window = row["window"]
        kept = list(window)
        if min_score is not None:
            kept = [w for w in kept if w["score"] >= float(min_score)]
        if score_margin is not None and window:
            best = float(window[0]["score"])
            floor = best - float(score_margin)
            kept = [w for w in kept if w["score"] >= floor]
        if not kept and window:
            kept = [max(window, key=lambda w: w["score"])]

        flags = [bool(w["is_gold"]) for w in kept]
        key_sets = [set(w["match_keys"]) for w in kept]
        ranked_ids = [w["chunk_id"] for w in kept if w.get("chunk_id")]

        per_case.append(
            {
                "recall@5": _recall_at_k_keys(key_sets, gold, 5),
                "precision@5": _precision_at_k_hits(flags, 5),
                "mrr": _mrr_hits(flags),
                "ndcg@10": ndcg_at_k(ranked_ids, list(gold), 10),
                "n_retrieved": float(len(kept)),
            }
        )

    averages = {
        "recall@5": round(mean([c["recall@5"] for c in per_case]), 4),
        "precision@5": round(mean([c["precision@5"] for c in per_case]), 4),
        "mrr": round(mean([c["mrr"] for c in per_case]), 4),
        "ndcg@10": round(mean([c["ndcg@10"] for c in per_case]), 4),
        "mean_n_retrieved": round(mean([c["n_retrieved"] for c in per_case]), 3),
    }
    return {
        "min_score": min_score,
        "score_margin": score_margin,
        "averages": averages,
        "n_cases": len(per_case),
    }


def score_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_scores: list[float] = []
    other_scores: list[float] = []
    for row in rows:
        for w in row["window"]:
            (gold_scores if w["is_gold"] else other_scores).append(float(w["score"]))

    def _summ(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"n": 0}
        vs = sorted(vals)
        return {
            "n": len(vs),
            "min": round(vs[0], 4),
            "p25": round(vs[len(vs) // 4], 4),
            "median": round(median(vs), 4),
            "p75": round(vs[(3 * len(vs)) // 4], 4),
            "max": round(vs[-1], 4),
            "mean": round(mean(vs), 4),
        }

    return {"gold_hit_scores": _summ(gold_scores), "non_hit_scores": _summ(other_scores)}


def pick_cutoff(dist: dict[str, Any], candidates: list[float]) -> float:
    gold = dist.get("gold_hit_scores") or {}
    other = dist.get("non_hit_scores") or {}
    if not gold.get("n") or not other.get("n"):
        return candidates[len(candidates) // 2]
    lo = float(other.get("median") or 0.0)
    hi = float(gold.get("p25") or gold.get("median") or lo)
    if hi < lo:
        hi = float(gold.get("median") or lo)
    target = (lo + hi) / 2.0
    return min(candidates, key=lambda c: abs(c - target))


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidates",
        # BGE cross-encoder logits on this corpus cluster ~0–45.
        default="0,10,20,25,26,26.5,27,28,28.5,29,30",
        help="Comma-separated RERANK_MIN_SCORE candidates",
    )
    p.add_argument(
        "--margins",
        default="0.5,1,1.5,2,3",
        help="Comma-separated RERANK_SCORE_MARGIN candidates (relative to best)",
    )
    p.add_argument("--top-n", type=int, default=int(os.getenv("RERANK_TOP_K", "5")))
    p.add_argument("--max-recall-drop", type=float, default=0.05)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "eval" / "results" / "rerank_min_score_ab.json",
    )
    args = p.parse_args(argv)
    candidates = [float(x.strip()) for x in args.candidates.split(",") if x.strip()]
    margins = [float(x.strip()) for x in args.margins.split(",") if x.strip()]

    logger.info("Collecting scored top-%d windows (cases with gold keys only)...", args.top_n)
    rows = collect_scored_windows(top_n=args.top_n)
    if not rows:
        logger.error("No golden cases with gold keys — abort")
        return 1

    dist = score_distribution(rows)
    baseline = apply_cutoff(rows, None)
    baseline["tag"] = "baseline_no_min_score"
    logger.info("baseline n=%d averages=%s", baseline["n_cases"], baseline["averages"])

    suggested = pick_cutoff(dist, candidates)
    logger.info("distribution=%s suggested=%s", dist, suggested)

    base_r = float(baseline["averages"]["recall@5"])
    base_p = float(baseline["averages"]["precision@5"])
    trials: list[dict[str, Any]] = []

    def _consider(row: dict[str, Any], best: dict[str, Any] | None) -> dict[str, Any] | None:
        r = float(row["averages"]["recall@5"])
        p = float(row["averages"]["precision@5"])
        row["delta_recall@5"] = round(r - base_r, 4)
        row["delta_precision@5"] = round(p - base_p, 4)
        trials.append(row)
        logger.info(
            "trial min=%s margin=%s recall@5=%.4f (Δ%+.4f) precision@5=%.4f (Δ%+.4f) mean_n=%.2f",
            row.get("min_score"),
            row.get("score_margin"),
            r,
            r - base_r,
            p,
            p - base_p,
            row["averages"]["mean_n_retrieved"],
        )
        if (r - base_r) < -abs(args.max_recall_drop):
            return best
        if p + 1e-9 < base_p:
            return best
        if best is None:
            return row
        bp = float(best["averages"]["precision@5"])
        br = float(best["averages"]["recall@5"])
        if p > bp or (abs(p - bp) < 1e-9 and r > br) or (
            abs(p - bp) < 1e-9
            and abs(r - br) < 1e-9
            and row["averages"]["mean_n_retrieved"] < best["averages"]["mean_n_retrieved"]
        ):
            return row
        return best

    best: dict[str, Any] | None = None
    for cut in sorted(set(candidates) | {suggested}):
        row = apply_cutoff(rows, cut)
        row["tag"] = f"min_score_{cut}"
        best = _consider(row, best)

    for margin in margins:
        row = apply_cutoff(rows, None, score_margin=margin)
        row["tag"] = f"margin_{margin}"
        best = _consider(row, best)

    if best is None or (
        float(best["averages"]["precision@5"]) <= base_p + 1e-9
        and float(best["averages"]["mean_n_retrieved"])
        >= baseline["averages"]["mean_n_retrieved"] - 1e-9
    ):
        shrinking = [
            t
            for t in trials
            if t["delta_recall@5"] >= -abs(args.max_recall_drop)
            and t["averages"]["precision@5"] + 1e-9 >= base_p
            and t["averages"]["mean_n_retrieved"] + 1e-9
            < baseline["averages"]["mean_n_retrieved"]
        ]
        if shrinking:
            best = max(
                shrinking,
                key=lambda t: (
                    t["averages"]["precision@5"],
                    t["averages"]["recall@5"],
                    -t["averages"]["mean_n_retrieved"],
                ),
            )

    if best is None:
        verdict = (
            "No candidate improved precision (or trimmed low-relevance hits) without "
            "exceeding recall drop budget; leave floors unset."
        )
        env_lines = ["# leave RERANK_MIN_SCORE / RERANK_SCORE_MARGIN unset\n"]
    else:
        verdict = (
            f"Select min_score={best.get('min_score')} margin={best.get('score_margin')}: "
            f"precision@5 {base_p:.4f}→{best['averages']['precision@5']} "
            f"(Δ{best['delta_precision@5']:+.4f}), "
            f"recall@5 {base_r:.4f}→{best['averages']['recall@5']} "
            f"(Δ{best['delta_recall@5']:+.4f}), "
            f"mean_n {baseline['averages']['mean_n_retrieved']}→"
            f"{best['averages']['mean_n_retrieved']}."
        )
        env_lines = []
        if best.get("min_score") is not None:
            env_lines.append(f"RERANK_MIN_SCORE={best['min_score']}\n")
        if best.get("score_margin") is not None:
            env_lines.append(f"RERANK_SCORE_MARGIN={best['score_margin']}\n")
        if not env_lines:
            env_lines = ["# leave unset\n"]

    report = {
        "baseline": baseline,
        "score_distribution": dist,
        "suggested_cutoff": suggested,
        "max_recall_drop": args.max_recall_drop,
        "gold_keyed_cases": len(rows),
        "case_ids": [r["id"] for r in rows],
        "trials": trials,
        "selected": best,
        "verdict": verdict,
        "stages_1_3_confirmation": str(
            ROOT / "data" / "vlm_figure_pass" / "stages_1_3_confirmation.json"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    rec = ROOT / "data" / "vlm_figure_pass" / "rerank_min_score.recommended"
    rec.write_text("".join(env_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "gold_keyed_cases": len(rows),
                "baseline": baseline["averages"],
                "selected": best,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
