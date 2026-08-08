"""Render a stakeholder RAGAS quality dashboard PNG (distinct from pass/fail).

Usage::

    python -m eval.render_ragas_dashboard eval/results/<run_id>/results.json
    python -m eval.render_ragas_dashboard --run-id 20260805T152635Z

Reads ``results.json`` from ``eval.run_full`` (or a standalone RAGAS-only payload
with ``cases[].ragas``). Writes ``eval/results/{run_id}/ragas_dashboard.png``.

Visually matched to ``eval.render_dashboard`` (same canvas, palette, fonts) so the
two PNGs read as a pair — this one is continuous 0–1 quality, not pass/fail.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval.scoring.ragas_scorer import (
    DEFAULT_RAGAS_JUDGE_MODEL,
    RAGAS_SCORE_CATEGORIES,
    _judge_model_name,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

# Match eval.render_dashboard canvas (1600×1000 px at 100 dpi).
FIG_WIDTH_IN = 16.0
FIG_HEIGHT_IN = 10.0
FIG_DPI = 100

METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}
METRIC_SHORT = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Relevancy",
    "context_precision": "Precision",
    "context_recall": "Recall",
}

# Stable display order (RAGAS-applicable categories only).
CATEGORY_ORDER = (
    "factual_lookup",
    "compliance_check",
    "multi_hop",
    "enumerative",
    "cross_regulation",
    "design_implication",
)

COLOR_HIGH = "#2E7D32"  # >= 0.8
COLOR_MID = "#F9A825"  # 0.5–0.79
COLOR_LOW = "#C62828"  # < 0.5
COLOR_EMPTY = "#BDBDBD"
COLOR_TEXT = "#212121"
COLOR_MUTED = "#616161"
COLOR_FOOTER = "#424242"
COLOR_BG = "#FAFAFA"
COLOR_RULE = "#E0E0E0"

CAPTION = (
    "This shows RAGAS quality scores (0-1). See dashboard.png for pass/fail rates."
)


def _format_run_date(results: dict[str, Any]) -> str:
    ts = results.get("timestamp") or results.get("run_date") or ""
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return str(ts) or "—"


def _finite_or_nan(raw: Any) -> float:
    if raw is None:
        return float("nan")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(val):
        return float("nan")
    return val


def _score_color(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return COLOR_EMPTY
    if value >= 0.8:
        return COLOR_HIGH
    if value >= 0.5:
        return COLOR_MID
    return COLOR_LOW


def _ragas_cases(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Cases that belong to RAGAS-scored categories (with or without scores)."""
    rows = results.get("cases") or results.get("results") or []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cat = str(row.get("category") or "").strip().lower()
        if cat in RAGAS_SCORE_CATEGORIES:
            out.append(row)
    return out


def compute_overall_ragas(
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Overall nanmean + coverage for each metric across RAGAS-applicable cases."""
    n_total = len(cases)
    out: dict[str, dict[str, Any]] = {}
    for key in METRIC_KEYS:
        vals = [
            _finite_or_nan((row.get("ragas") or {}).get(key))
            for row in cases
        ]
        arr = np.asarray(vals, dtype=float)
        n_scored = int(np.sum(~np.isnan(arr)))
        avg = float(np.nanmean(arr)) if n_scored else float("nan")
        if n_scored:
            avg = round(avg, 4)
        out[key] = {
            "average": None if math.isnan(avg) else avg,
            "n_scored": n_scored,
            "n_cases": n_total,
            "n_nan": n_total - n_scored,
            "display": (
                f"{avg:.2f} ({n_scored}/{n_total} cases scored)"
                if n_scored
                else f"n/a (0/{n_total} cases scored)"
            ),
        }
    return out


def compute_per_category_ragas(
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-category nanmean + coverage for each RAGAS metric."""
    buckets: dict[str, list[dict[str, Any]]] = {c: [] for c in CATEGORY_ORDER}
    for row in cases:
        cat = str(row.get("category") or "").strip().lower()
        if cat in buckets:
            buckets[cat].append(row)

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cat, rows in buckets.items():
        if not rows:
            # Still list the category shell so stakeholders see the full roster.
            out[cat] = {
                key: {
                    "average": None,
                    "n_scored": 0,
                    "n_cases": 0,
                    "n_nan": 0,
                }
                for key in METRIC_KEYS
            }
            continue
        out[cat] = {}
        n = len(rows)
        for key in METRIC_KEYS:
            vals = [_finite_or_nan((r.get("ragas") or {}).get(key)) for r in rows]
            arr = np.asarray(vals, dtype=float)
            n_scored = int(np.sum(~np.isnan(arr)))
            avg = float(np.nanmean(arr)) if n_scored else float("nan")
            out[cat][key] = {
                "average": None if math.isnan(avg) else round(avg, 4),
                "n_scored": n_scored,
                "n_cases": n,
                "n_nan": n - n_scored,
            }
    return out


def resolve_judge_model(results: dict[str, Any]) -> str:
    """Best-effort judge label from payload, case rows, or env/defaults."""
    for key in ("judge_model", "ragas_judge_model", "ragas_judge"):
        val = results.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for row in results.get("cases") or []:
        if not isinstance(row, dict):
            continue
        for key in ("judge_model", "ragas_judge_model"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    try:
        return _judge_model_name()
    except Exception:  # noqa: BLE001
        return (
            (os.getenv("RAGAS_JUDGE_MODEL") or "").strip()
            or (os.getenv("EVAL_JUDGE_MODEL") or "").strip()
            or DEFAULT_RAGAS_JUDGE_MODEL
        )


def render_ragas_dashboard(
    results: dict[str, Any],
    *,
    out_path: Path | None = None,
) -> Path:
    """Render ``ragas_dashboard.png`` from a results (or RAGAS-only) payload."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch

    cases = _ragas_cases(results)
    overall = compute_overall_ragas(cases)
    per_cat = compute_per_category_ragas(cases)
    judge = resolve_judge_model(results)
    run_date = _format_run_date(results)
    run_id = str(results.get("run_id") or "latest")
    n_ragas = len(cases)
    n_total_cases = int(
        results.get("n_cases")
        or results.get("n_golden")
        or len(results.get("cases") or [])
        or n_ragas
    )

    if out_path is None:
        out_path = RESULTS_DIR / run_id / "ragas_dashboard.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=FIG_DPI)
    fig.patch.set_facecolor(COLOR_BG)

    # --- Caption (distinguish from pass/fail dashboard) ---
    fig.text(
        0.5,
        0.975,
        CAPTION,
        fontsize=11,
        fontstyle="italic",
        color=COLOR_MUTED,
        ha="center",
        va="top",
    )

    # --- Header ---
    fig.text(
        0.05,
        0.935,
        "AutoSafety RAG — RAGAS Quality Scores",
        fontsize=18,
        fontweight="bold",
        color=COLOR_TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.05,
        0.895,
        f"{run_date}   ·   {n_total_cases} cases "
        f"({n_ragas} RAGAS-applicable)   ·   run {run_id}",
        fontsize=12,
        color=COLOR_MUTED,
        ha="left",
        va="top",
    )
    fig.text(
        0.95,
        0.925,
        "RAGAS QUALITY (0–1)",
        fontsize=16,
        fontweight="bold",
        color="#1565C0",
        ha="right",
        va="top",
    )

    # --- Overall summary cards ---
    card_y = 0.72
    card_h = 0.14
    card_w = 0.21
    gap = 0.02
    left0 = 0.05
    for i, key in enumerate(METRIC_KEYS):
        x = left0 + i * (card_w + gap)
        info = overall[key]
        avg = info["average"]
        color = _score_color(avg)
        fig.patches.append(
            FancyBboxPatch(
                (x, card_y),
                card_w,
                card_h,
                transform=fig.transFigure,
                boxstyle="round,pad=0.012,rounding_size=0.01",
                facecolor="#FFFFFF",
                edgecolor=color,
                linewidth=2.2,
                zorder=1,
            )
        )
        fig.text(
            x + card_w / 2,
            card_y + card_h - 0.028,
            METRIC_LABELS[key],
            fontsize=11,
            color=COLOR_MUTED,
            ha="center",
            va="top",
            zorder=2,
        )
        score_txt = f"{avg:.2f}" if avg is not None else "n/a"
        fig.text(
            x + card_w / 2,
            card_y + card_h / 2 + 0.005,
            score_txt,
            fontsize=28,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            zorder=2,
        )
        fig.text(
            x + card_w / 2,
            card_y + 0.022,
            f"{info['n_scored']}/{info['n_cases']} cases scored",
            fontsize=10,
            color=COLOR_FOOTER,
            ha="center",
            va="bottom",
            zorder=2,
        )

    # --- Per-category grouped horizontal bars ---
    ax = fig.add_axes([0.22, 0.16, 0.70, 0.50])
    ax.set_facecolor(COLOR_BG)

    cats = list(CATEGORY_ORDER)
    n_metrics = len(METRIC_KEYS)
    group_height = 0.78
    bar_height = group_height / n_metrics
    y_centers = list(range(len(cats)))

    for gi, cat in enumerate(cats):
        metrics = per_cat.get(cat) or {}
        base = gi - group_height / 2 + bar_height / 2
        for mi, key in enumerate(METRIC_KEYS):
            info = metrics.get(key) or {}
            avg = info.get("average")
            width = float(avg) if avg is not None else 0.0
            y = base + mi * bar_height
            color = _score_color(avg)
            ax.barh(
                y,
                width if avg is not None else 0.02,
                height=bar_height * 0.85,
                color=color,
                edgecolor="none",
                alpha=1.0 if avg is not None else 0.35,
                zorder=2,
            )
            if avg is None:
                label = "n/a"
                lx = 0.03
            else:
                n_scored = int(info.get("n_scored") or 0)
                n_cases = int(info.get("n_cases") or 0)
                label = f"{avg:.2f} ({n_scored}/{n_cases})"
                lx = min(avg + 0.015, 0.98) if avg < 0.92 else avg - 0.015
            ax.text(
                lx,
                y,
                label,
                va="center",
                ha="left" if avg is None or avg < 0.92 else "right",
                fontsize=8,
                color=COLOR_TEXT if (avg is None or avg < 0.92) else "#FFFFFF",
                zorder=4,
            )

    ax.set_yticks(y_centers)
    ax.set_yticklabels(cats, fontsize=11, color=COLOR_TEXT)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("RAGAS score (0–1)", fontsize=11, color=COLOR_FOOTER)
    ax.axvline(0.8, color="#BDBDBD", linestyle="--", linewidth=0.8, zorder=1)
    ax.axvline(0.5, color="#BDBDBD", linestyle=":", linewidth=0.8, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()  # factual_lookup at top

    # Metric order key (bars are score-colored; order identifies the metric)
    fig.text(
        0.05,
        0.62,
        "Per category (4 bars each):\n"
        + "\n".join(f"  {i + 1}. {METRIC_SHORT[k]}" for i, k in enumerate(METRIC_KEYS)),
        fontsize=9,
        color=COLOR_MUTED,
        ha="left",
        va="top",
        family="monospace",
    )

    legend_patches = [
        Patch(facecolor=COLOR_HIGH, label="≥ 0.80"),
        Patch(facecolor=COLOR_MID, label="0.50 – 0.79"),
        Patch(facecolor=COLOR_LOW, label="< 0.50"),
        Patch(facecolor=COLOR_EMPTY, label="no score"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        frameon=False,
        fontsize=9,
        title="Score band",
        title_fontsize=9,
    )

    # --- Footer ---
    fig.lines.append(
        plt.Line2D(
            [0.05, 0.95],
            [0.09, 0.09],
            transform=fig.transFigure,
            color=COLOR_RULE,
            linewidth=1,
        )
    )
    footer = (
        f"Judged by: {judge} — a different model family than the models being "
        "evaluated, to reduce judge bias"
    )
    try:
        from eval.retrieval_eval import (
            compute_retrieval_quality_report,
            format_retrieval_quality_footer,
        )

        rq = compute_retrieval_quality_report(
            list(results.get("cases") or results.get("results") or [])
        )
        results["retrieval_quality"] = rq
        footer += "  ·  " + format_retrieval_quality_footer(rq)
    except Exception as exc:  # noqa: BLE001
        logger.debug("retrieval quality footer skipped: %s", exc)
    fig.text(
        0.5,
        0.05,
        footer,
        fontsize=9 if "Retrieval quality" in footer else 11,
        color=COLOR_FOOTER,
        ha="center",
        va="center",
        wrap=True,
    )

    fig.savefig(out_path, dpi=FIG_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(
        "Wrote RAGAS dashboard %s (%sx%s)",
        out_path,
        int(FIG_WIDTH_IN * FIG_DPI),
        int(FIG_HEIGHT_IN * FIG_DPI),
    )
    return out_path


def render_ragas_dashboard_from_path(
    results_path: Path, *, out_path: Path | None = None
) -> Path:
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    if out_path is None:
        out_path = Path(results_path).resolve().parent / "ragas_dashboard.png"
    return render_ragas_dashboard(data, out_path=out_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render RAGAS quality dashboard PNG")
    p.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=None,
        help="Path to results.json (default: path in eval/results/latest.json)",
    )
    p.add_argument("--run-id", default=None, help="Load eval/results/{run_id}/results.json")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.run_id:
        results_path = RESULTS_DIR / args.run_id / "results.json"
    elif args.results:
        results_path = args.results
    else:
        latest = RESULTS_DIR / "latest.json"
        if not latest.is_file():
            print("No results path / --run-id / eval/results/latest.json", flush=True)
            return 2
        meta = json.loads(latest.read_text(encoding="utf-8"))
        results_path = Path(meta.get("results_path") or meta.get("path") or "")
        if not results_path.is_file() and meta.get("run_id"):
            results_path = RESULTS_DIR / str(meta["run_id"]) / "results.json"

    if not Path(results_path).is_file():
        print(f"results.json not found: {results_path}", flush=True)
        return 2

    out = render_ragas_dashboard_from_path(Path(results_path), out_path=args.out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
