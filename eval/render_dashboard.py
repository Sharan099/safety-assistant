"""Render a stakeholder-ready eval dashboard PNG from results.json or partial_results.jsonl.

Defaults assume the canonical **30-case** ``eval/golden_set.jsonl``. Historical
pre-30 runs live under ``eval/results_archive_pre30/`` (pass that path explicitly).

Usage::

    python -m eval.render_dashboard eval/results/<run_id>/results.json
    python -m eval.render_dashboard --run-id 20260805T120000Z
    python -m eval.render_dashboard --run-id RUN_ID --partial
    python -m eval.render_dashboard eval/results_archive_pre30/<run_id>/results.json

Called automatically at the end of ``eval.run_full`` (full results). Use ``--partial``
to preview an in-progress run from ``partial_results.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

# Always outline these even if thresholds.yaml severity is missing.
DEFAULT_CRITICAL_CATEGORIES = frozenset({"numeric_safety", "prompt_injection"})

# Slide-friendly canvas (1600×1000 px at 100 dpi).
FIG_WIDTH_IN = 16.0
FIG_HEIGHT_IN = 10.0
FIG_DPI = 100

PENDING_BAR_COLOR = "#BDBDBD"
PENDING_LABEL = "not yet evaluated"
PARTIAL_BANNER = "PARTIAL RESULTS — {n}/{total} cases completed"


def _pass_rate_pct(agg: dict[str, Any]) -> float | None:
    """Return pass rate as 0–100, or None when the category has not been evaluated."""
    if agg.get("pending") or int(agg.get("n_cases") or 0) == 0 and agg.get("pass_rate") is None:
        return None
    pr = agg.get("pass_rate")
    if pr is None:
        return None
    val = float(pr)
    if val <= 1.0:
        return max(0.0, min(100.0, val * 100.0))
    return max(0.0, min(100.0, val))


def _bar_color(pct: float | None, *, pending: bool = False) -> str:
    if pending or pct is None:
        return PENDING_BAR_COLOR
    if pct >= 90.0:
        return "#2E7D32"
    if pct >= 70.0:
        return "#F9A825"
    return "#C62828"


def _status_color(status: str) -> str:
    s = (status or "").strip().upper()
    if s.startswith("PARTIAL"):
        return "#E65100"
    if s == "PRODUCTION READY":
        return "#2E7D32"
    if s == "NOT PRODUCTION READY":
        return "#C62828"
    if s == "NEEDS IMPROVEMENT":
        return "#F9A825"
    return "#424242"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = float(seconds)
    if s >= 3600:
        return f"{s / 3600:.2f} h"
    if s >= 60:
        return f"{s / 60:.1f} min"
    return f"{s:.1f} s"


def _format_run_date(results: dict[str, Any]) -> str:
    ts = results.get("timestamp") or ""
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return str(ts) or "—"


def _critical_categories(results: dict[str, Any]) -> set[str]:
    crit = set(DEFAULT_CRITICAL_CATEGORIES)
    gate_cats = (results.get("gate") or {}).get("categories") or {}
    for name, info in gate_cats.items():
        if str((info or {}).get("severity") or "").lower() == "critical":
            crit.add(str(name).strip().lower())
    thr_cats = (results.get("thresholds_categories") or {})
    for name, info in thr_cats.items():
        if str((info or {}).get("severity") or "").lower() == "critical":
            crit.add(str(name).strip().lower())
    return crit


def _category_rows(
    results: dict[str, Any],
) -> list[tuple[str, float | None, bool, bool]]:
    """Return (category, pass_rate_pct|None, is_critical, pending).

    Evaluated categories sort lowest pass rate first; pending categories follow
    (alphabetically) so a grey \"not yet evaluated\" bar is never confused with 0%.
    """
    per_cat = results.get("per_category") or {}
    critical = _critical_categories(results)
    evaluated: list[tuple[str, float | None, bool, bool]] = []
    pending: list[tuple[str, float | None, bool, bool]] = []
    for name, agg in per_cat.items():
        cat = str(name).strip().lower()
        agg_d = agg or {}
        is_pending = bool(agg_d.get("pending")) or (
            int(agg_d.get("n_cases") or 0) == 0 and agg_d.get("pass_rate") is None
        )
        pct = None if is_pending else _pass_rate_pct(agg_d)
        row = (cat, pct, cat in critical, is_pending)
        if is_pending:
            pending.append(row)
        else:
            evaluated.append(row)
    evaluated.sort(key=lambda r: (r[1] if r[1] is not None else 999.0, r[0]))
    pending.sort(key=lambda r: r[0])
    return evaluated + pending


def _sum_partial_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_cost = 0.0
    total_tokens = 0
    wall = 0.0
    for row in rows:
        cost = row.get("_case_cost") or {}
        if isinstance(cost, dict):
            total_cost += float(cost.get("cost_usd") or 0.0)
            total_tokens += int(cost.get("input_tokens") or 0) + int(
                cost.get("output_tokens") or 0
            )
        timing = row.get("_case_timing_seconds")
        if timing is not None:
            try:
                wall += float(timing)
            except (TypeError, ValueError):
                pass
    return {
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "wall_clock_seconds": round(wall, 3) if wall else None,
    }


def _expected_categories_and_total(
    run_dir: Path,
    case_rows: list[dict[str, Any]],
) -> tuple[list[str], int]:
    """Resolve the full category roster + planned case count for a partial run."""
    from eval.aggregation import DEFAULT_THRESHOLDS, load_thresholds
    from eval.gold import DEFAULT_GOLDEN, load_golden_set

    config_path = run_dir / "run_config.json"
    run_config: dict[str, Any] = {}
    if config_path.is_file():
        run_config = json.loads(config_path.read_text(encoding="utf-8"))

    case_ids = list(run_config.get("case_ids") or [])
    n_total = len(case_ids) if case_ids else max(len(case_rows), 0)

    gold_path = Path(run_config["gold_path"]) if run_config.get("gold_path") else DEFAULT_GOLDEN
    thr_path = (
        Path(run_config["thresholds_path"])
        if run_config.get("thresholds_path")
        else DEFAULT_THRESHOLDS
    )

    cats: set[str] = set()
    if case_ids:
        try:
            by_id = {
                str(c.get("id")): c for c in load_golden_set(gold_path) if c.get("id") is not None
            }
            for cid in case_ids:
                case = by_id.get(str(cid))
                if case:
                    cat = str(case.get("category") or "").strip().lower()
                    if cat:
                        cats.add(cat)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not map run_config case_ids via golden set: %s", exc)

    try:
        thr = load_thresholds(thr_path)
        for name in (thr.get("categories") or {}):
            cats.add(str(name).strip().lower())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load thresholds for category roster: %s", exc)

    for row in case_rows:
        cat = str(row.get("category") or "").strip().lower()
        if cat:
            cats.add(cat)

    if not n_total:
        n_total = len(case_rows)
    return sorted(cats), n_total


def build_results_from_partial(
    run_dir: Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a dashboard payload from ``partial_results.jsonl`` (+ optional run_config).

    Present categories use the same strict per-category aggregation as a finished
    run. Categories with zero completed cases are marked ``pending`` so the
    renderer shows grey \"not yet evaluated\" instead of a misleading 0% red bar.
    """
    from eval.aggregation import (
        DEFAULT_THRESHOLDS,
        aggregate_per_category,
        load_partial_results,
        load_thresholds,
    )

    run_dir = Path(run_dir)
    partial_path = run_dir / "partial_results.jsonl"
    if not partial_path.is_file():
        raise FileNotFoundError(f"partial_results.jsonl not found: {partial_path}")

    rows = load_partial_results(partial_path)
    expected_cats, n_total = _expected_categories_and_total(run_dir, rows)
    per_category = aggregate_per_category(rows)

    for cat in expected_cats:
        if cat not in per_category:
            per_category[cat] = {
                "n_cases": 0,
                "n_pass": 0,
                "pass_rate": None,
                "pending": True,
            }

    # Severity hints for critical outlines (gate not applied on partial).
    thr_cats: dict[str, Any] = {}
    config_path = run_dir / "run_config.json"
    thr_path = DEFAULT_THRESHOLDS
    if config_path.is_file():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if cfg.get("thresholds_path"):
            thr_path = Path(cfg["thresholds_path"])
    try:
        thr_cats = (load_thresholds(thr_path).get("categories") or {})
    except Exception:  # noqa: BLE001
        thr_cats = {}

    rid = run_id or run_dir.name
    n_done = len(rows)
    return {
        "run_id": rid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "partial": True,
        "n_cases": n_done,
        "n_cases_completed": n_done,
        "n_cases_total": n_total,
        "overall_status": f"PARTIAL — {n_done}/{n_total} (not a final verdict)",
        "per_category": per_category,
        "thresholds_categories": thr_cats,
        "gate": {"categories": {k: {"severity": (v or {}).get("severity")} for k, v in thr_cats.items()}},
        "cost_summary": _sum_partial_cost(rows),
        "cases": rows,
    }


def render_dashboard(
    results: dict[str, Any],
    *,
    out_path: Path | None = None,
) -> Path:
    """Render ``dashboard.png`` from a results.json (or partial-built) payload."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    rows = _category_rows(results)
    n_cats = len(rows)
    is_partial = bool(results.get("partial"))
    n_completed = int(
        results.get("n_cases_completed")
        if results.get("n_cases_completed") is not None
        else results.get("n_cases")
        or len(results.get("cases") or [])
    )
    n_total = int(results.get("n_cases_total") or n_completed)
    n_questions = n_completed
    status = str(
        results.get("overall_status")
        or (results.get("gate") or {}).get("overall_status")
        or "—"
    )
    cost = results.get("cost_summary") or {}
    run_date = _format_run_date(results)

    if out_path is None:
        rid = results.get("run_id") or "latest"
        out_path = RESULTS_DIR / str(rid) / "dashboard.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=FIG_DPI)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Top layout shifts down when the partial banner is present.
    header_y = 0.90 if is_partial else 0.955
    sub_y = 0.86 if is_partial else 0.915
    status_y = 0.89 if is_partial else 0.945
    plot_top = 0.64 if is_partial else 0.70

    if is_partial:
        banner = PARTIAL_BANNER.format(n=n_completed, total=n_total)
        # Full-width warning strip — unmistakable this is not a final gate.
        fig.patches.append(
            FancyBboxPatch(
                (0.0, 0.935),
                1.0,
                0.065,
                transform=fig.transFigure,
                boxstyle="square,pad=0",
                facecolor="#E65100",
                edgecolor="none",
                zorder=10,
            )
        )
        fig.text(
            0.5,
            0.967,
            banner,
            fontsize=16,
            fontweight="bold",
            color="#FFFFFF",
            ha="center",
            va="center",
            zorder=11,
        )

    # Header
    fig.text(
        0.05,
        header_y,
        "AutoSafety RAG — Evaluation Readiness",
        fontsize=18,
        fontweight="bold",
        color="#212121",
        ha="left",
        va="top",
    )
    rid = str(results.get("run_id") or "latest")
    meta = f"{run_date}   ·   {n_questions} questions   ·   run {rid}"
    if is_partial:
        meta += f" of {n_total} planned"
    meta += f"   ·   {n_cats} categories"
    fig.text(
        0.05,
        sub_y,
        meta,
        fontsize=12,
        color="#616161",
        ha="left",
        va="top",
    )
    fig.text(
        0.95,
        status_y,
        status,
        fontsize=16 if is_partial else 22,
        fontweight="bold",
        color=_status_color(status),
        ha="right",
        va="top",
    )

    ax.set_position([0.30, 0.14, 0.62, plot_top])

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No category results to display",
            ha="center",
            va="center",
            fontsize=14,
            color="#757575",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        rates = [0.0 if r[3] else (r[1] if r[1] is not None else 0.0) for r in rows]
        # Pending: show a full-width light grey track so the state is visible.
        display_widths = [100.0 if r[3] else rates[i] for i, r in enumerate(rows)]
        colors = [_bar_color(r[1], pending=r[3]) for r in rows]
        y_pos = list(range(len(rows)))

        bars = ax.barh(
            y_pos,
            display_widths,
            height=0.65,
            color=colors,
            edgecolor="none",
            zorder=2,
        )
        for i, r in enumerate(rows):
            bars[i].set_alpha(0.45 if r[3] else 1.0)

        for i, (cat, pct, is_crit, is_pending) in enumerate(rows):
            if is_crit and not is_pending:
                bars[i].set_edgecolor("#000000")
                bars[i].set_linewidth(2.8)
                bars[i].set_zorder(3)

            if is_pending:
                ax.text(
                    50.0,
                    i,
                    PENDING_LABEL,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontstyle="italic",
                    color="#424242",
                    zorder=4,
                )
            else:
                assert pct is not None
                label_x = min(pct + 1.5, 99.0) if pct < 97 else pct - 1.5
                ax.text(
                    label_x,
                    i,
                    f"{pct:.0f}%",
                    va="center",
                    ha="left" if pct < 97 else "right",
                    fontsize=10,
                    color="#212121" if pct < 97 else "#FFFFFF",
                    zorder=4,
                )

        ax.set_yticks(y_pos)
        tick_labels = []
        for cat, _pct, is_crit, is_pending in rows:
            if is_pending:
                tick_labels.append(f"{cat}   (pending)")
            elif is_crit:
                tick_labels.append(f"{cat}   CRITICAL")
            else:
                tick_labels.append(cat)
        yticks = ax.set_yticklabels(tick_labels, fontsize=11, color="#212121")
        for tick, (_cat, _pct, is_crit, is_pending) in zip(yticks, rows):
            if is_pending:
                tick.set_color("#757575")
                tick.set_fontstyle("italic")
            elif is_crit:
                tick.set_fontweight("bold")

        ax.set_xlim(0, 100)
        ax.set_xlabel("Pass rate (%)", fontsize=11, color="#424242")
        ax.axvline(90, color="#BDBDBD", linestyle="--", linewidth=0.8, zorder=1)
        ax.axvline(70, color="#BDBDBD", linestyle=":", linewidth=0.8, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BDBDBD")
        ax.spines["bottom"].set_color("#BDBDBD")
        ax.tick_params(axis="x", colors="#616161")
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.8)

    # Footer
    total_cost = float(cost.get("total_cost_usd") or 0.0)
    total_tokens = int(cost.get("total_tokens") or 0)
    duration = _format_duration(cost.get("wall_clock_seconds"))
    footer = (
        f"Eval cost  ${total_cost:.4f}"
        f"    ·    Tokens  {total_tokens:,}"
        f"    ·    Duration  {duration}"
    )
    if is_partial:
        footer += "    ·    PARTIAL RUN — not a production-readiness verdict"
    try:
        from eval.retrieval_eval import (
            compute_retrieval_quality_report,
            format_retrieval_quality_footer,
        )

        rq = compute_retrieval_quality_report(
            list(results.get("cases") or results.get("results") or [])
        )
        results["retrieval_quality"] = rq
        footer += "\n" + format_retrieval_quality_footer(rq)
    except Exception as exc:  # noqa: BLE001
        logger.debug("retrieval quality footer skipped: %s", exc)
    fig.add_artist(
        plt.Line2D(
            [0.05, 0.95],
            [0.08, 0.08],
            transform=fig.transFigure,
            color="#E0E0E0",
            linewidth=1,
        )
    )
    fig.text(
        0.5,
        0.045,
        footer,
        fontsize=9 if "\n" in footer else (11 if is_partial else 12),
        color="#424242",
        ha="center",
        va="center",
        linespacing=1.35,
    )

    fig.savefig(out_path, dpi=FIG_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(
        "Wrote dashboard %s (%sx%s)%s",
        out_path,
        int(FIG_WIDTH_IN * FIG_DPI),
        int(FIG_HEIGHT_IN * FIG_DPI),
        " [partial]" if is_partial else "",
    )
    return out_path


def render_dashboard_from_path(
    results_path: Path, *, out_path: Path | None = None
) -> Path:
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    if out_path is None:
        out_path = Path(results_path).resolve().parent / "dashboard.png"
    return render_dashboard(data, out_path=out_path)


def render_dashboard_from_partial(
    run_dir: Path,
    *,
    run_id: str | None = None,
    out_path: Path | None = None,
) -> Path:
    """Render a dashboard from ``eval/results/{run_id}/partial_results.jsonl``."""
    run_dir = Path(run_dir)
    data = build_results_from_partial(run_dir, run_id=run_id or run_dir.name)
    if out_path is None:
        out_path = run_dir / "dashboard.png"
    return render_dashboard(data, out_path=out_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render eval readiness dashboard PNG")
    p.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=None,
        help="Path to results.json (default: path in eval/results/latest.json)",
    )
    p.add_argument("--run-id", default=None, help="Load eval/results/{run_id}/…")
    p.add_argument(
        "--partial",
        action="store_true",
        help="Read partial_results.jsonl (in-progress run) instead of results.json",
    )
    p.add_argument("--out", type=Path, default=None, help="Output PNG path")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.partial:
        if args.run_id:
            run_dir = RESULTS_DIR / args.run_id
        elif args.results:
            # Allow passing the partial_results.jsonl path or its parent dir.
            rp = Path(args.results)
            run_dir = rp.parent if rp.name == "partial_results.jsonl" else rp
        else:
            print(
                "--partial requires --run-id RUN_ID (or a path to partial_results.jsonl)",
                flush=True,
            )
            return 2
        try:
            out = render_dashboard_from_partial(
                run_dir, run_id=args.run_id or run_dir.name, out_path=args.out
            )
        except FileNotFoundError as exc:
            print(str(exc), flush=True)
            return 2
        print(f"Wrote {out}")
        return 0

    if args.run_id:
        results_path = RESULTS_DIR / args.run_id / "results.json"
    elif args.results:
        results_path = args.results
    else:
        latest = RESULTS_DIR / "latest.json"
        if not latest.is_file():
            print("No results path given and eval/results/latest.json missing", flush=True)
            return 2
        meta = json.loads(latest.read_text(encoding="utf-8"))
        results_path = Path(
            meta.get("path") or (RESULTS_DIR / meta["run_id"] / "results.json")
        )

    if not results_path.is_file():
        print(f"results.json not found: {results_path}", flush=True)
        return 2

    out = render_dashboard_from_path(results_path, out_path=args.out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
