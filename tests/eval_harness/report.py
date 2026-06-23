"""Build combined evaluation reports (markdown + JSON)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import math

from config import EVALUATION_CURRENT


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    return obj


def format_scorecard_table(rows: list[dict]) -> str:
    lines = [
        "| id | type | recall | must_not | behavior | contains | forbidden | PASS/FAIL |",
        "|----|------|--------|----------|----------|----------|-----------|-----------|",
    ]
    for r in rows:
        status = "PASS" if r.get("pass") else "FAIL"
        lines.append(
            f"| {r['id']} | {r.get('query_type', '')} | {r.get('recall')} | "
            f"{r.get('must_not')} | {r.get('behavior')} | {r.get('contains')} | "
            f"{r.get('forbidden')} | **{status}** |"
        )
    return "\n".join(lines)


def top_failures(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if not r.get("pass"):
            out.append({
                "id": r["id"],
                "query_type": r.get("query_type"),
                "failures": r.get("failures", []),
                "details": r.get("details", {}),
            })
    return out


def build_eval_report(
    deterministic: dict[str, Any],
    ragas: dict[str, Any] | None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = deterministic.get("items", [])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        "deterministic": deterministic,
        "ragas": ragas,
        "top_failures": top_failures(rows),
    }


def write_eval_report(
    report: dict[str, Any],
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    out_dir = out_dir or EVALUATION_CURRENT
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "eval_report.json"
    md_path = out_dir / "eval_report.md"

    json_path.write_text(
        json.dumps(_json_sanitize(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    det = report.get("deterministic", {})
    agg = det.get("aggregate", {})
    rows = det.get("items", [])
    ragas = report.get("ragas") or {}
    token = (ragas.get("token_summary") or {}) if ragas else {}
    failures = report.get("top_failures", [])

    md_parts = [
        "# PSA AI Passive Safety — Evaluation Report",
        "",
        f"Generated: {report.get('generated_at', '')}",
        "",
        "## Stage A — Deterministic (zero tokens)",
        "",
        f"**Overall pass rate:** {agg.get('overall', {}).get('pass', 0)}/"
        f"{agg.get('overall', {}).get('total', 0)} "
        f"({agg.get('overall', {}).get('rate', 0):.1%})",
        "",
        "### Pass rate by query_type",
        "",
    ]

    for qt, stats in (agg.get("by_query_type") or {}).items():
        md_parts.append(f"- **{qt}**: {stats['pass']}/{stats['total']} ({stats['rate']:.1%})")

    md_parts.extend([
        "",
        "### Per-question scorecard",
        "",
        format_scorecard_table(rows),
        "",
    ])

    gw = (report.get("meta") or {}).get("gateway_tier_stats") or {}
    if gw.get("samples_with_model"):
        md_parts.extend([
            "## Gateway tier routing",
            "",
            f"- **Fast-tier (8B) rate:** {gw.get('fast_tier_rate', 0):.1%} "
            f"({gw.get('fast_tier_count', 0)}/{gw.get('samples_with_model', 0)})",
            f"- **Failover rate:** {gw.get('fallback_rate', 0):.1%}",
            f"- **Model distribution:** `{gw.get('model_distribution', {})}`",
            f"- **Regression gate:** {'PASS' if gw.get('gate_pass') else 'FAIL'} "
            f"(fast-tier rate must stay below 50%)",
            "",
        ])

    md_parts.extend([
        "## Stage B — RAGAS (budgeted)",
        "",
    ])

    if ragas:
        ar = ragas.get("answer_relevancy", {})
        jm = ragas.get("judge_metrics", {})
        md_parts.extend([
            f"- **answer_relevancy** (local embeddings, all non-abstention): "
            f"mean={ar.get('mean')} — {ar.get('note', '')}",
            f"- **faithfulness** (judge subset): mean={jm.get('faithfulness_mean')}",
            f"- **context_precision** (judge subset): mean={jm.get('context_precision_mean')}",
            f"- Judge subset IDs: `{', '.join(jm.get('subset_ids', []))}`",
            f"- Excluded abstention IDs: `{', '.join(ragas.get('excluded_from_ragas', []))}`",
            f"- Not run via RAGAS: `{', '.join(ragas.get('not_run_via_ragas', []))}` "
            f"— {ragas.get('not_run_note', '')}",
            "",
        ])
        if token:
            md_parts.extend([
                "### Token + call summary",
                "",
                f"- Groq judge calls: {token.get('groq_calls', 0)}",
                f"- Prompt tokens: {token.get('prompt_tokens', 0)}",
                f"- Completion tokens: {token.get('completion_tokens', 0)}",
                f"- Total tokens: {token.get('total_tokens', 0)} / budget {token.get('budget', 0)}",
                f"- Stopped on budget: {token.get('stopped_on_budget', False)}",
                f"- Skipped (rate limit): {token.get('skipped_rate_limit', [])}",
                "",
            ])
        skipped = []
        for iid, scores in (jm.get("per_id") or {}).items():
            for metric, val in scores.items():
                if val == "skipped_rate_limit":
                    skipped.append(f"{iid}/{metric}")
        if skipped:
            md_parts.append(f"- Rate-limit skips: {', '.join(skipped)}")
            md_parts.append("")
    else:
        md_parts.append("_RAGAS stage skipped._")
        md_parts.append("")

    md_parts.extend([
        "## Top failures",
        "",
    ])
    if failures:
        for f in failures:
            md_parts.append(
                f"- **{f['id']}** ({f.get('query_type')}): failed checks — "
                f"{', '.join(f.get('failures', []))}"
            )
    else:
        md_parts.append("_No failures — all items passed Stage A._")

    md_path.write_text("\n".join(md_parts) + "\n", encoding="utf-8")
    return json_path, md_path


def plot_eval_charts(
    report: dict[str, Any],
    out_dir: Path | None = None,
) -> list[Path]:
    """Write RAGAS + deterministic PNG dashboards to evaluation/current."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = out_dir or EVALUATION_CURRENT
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    ragas = report.get("ragas") or {}
    det = report.get("deterministic", {})
    agg = det.get("aggregate", {})

    # --- RAGAS metrics bar chart ---
    ar = ragas.get("answer_relevancy", {})
    jm = ragas.get("judge_metrics", {})
    metric_vals: list[tuple[str, float | None]] = [
        ("Answer Relevancy\n(local)", ar.get("mean")),
        ("Faithfulness\n(judge)", jm.get("faithfulness_mean")),
        ("Context Precision\n(judge)", jm.get("context_precision_mean")),
    ]
    labels = [m[0] for m in metric_vals]
    vals = [float(m[1]) if m[1] is not None else 0.0 for m in metric_vals]
    has_vals = any(m[1] is not None for m in metric_vals)

    if has_vals:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#3b82f6", "#22c55e", "#8b5cf6"]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score (0–1)")
        ax.set_title("RAGAS Metrics — PSA Passive Safety Golden Set (15Q)")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.02,
                    f"{v:.3f}",
                    ha="center",
                    fontsize=11,
                )
        subset = jm.get("subset_ids") or []
        if subset:
            ax.text(
                0.02,
                0.02,
                f"Judge subset: {', '.join(subset)}",
                transform=ax.transAxes,
                fontsize=8,
                color="#64748b",
            )
        fig.tight_layout()
        ragas_path = out_dir / "eval_ragas_metrics.png"
        fig.savefig(ragas_path, dpi=150)
        plt.close(fig)
        written.append(ragas_path)

    # --- Deterministic pass rate by query_type ---
    by_type = agg.get("by_query_type") or {}
    if by_type:
        types = list(by_type.keys())
        rates = [by_type[t]["rate"] for t in types]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(types, rates, color="#0ea5e9")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Pass rate")
        ax.set_title("Stage A — Deterministic Pass Rate by Query Type")
        for bar, r in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                r + 0.02,
                f"{r:.0%}",
                ha="center",
                fontsize=10,
            )
        fig.tight_layout()
        det_path = out_dir / "eval_deterministic_pass_rate.png"
        fig.savefig(det_path, dpi=150)
        plt.close(fig)
        written.append(det_path)

    # --- Overall scorecard image ---
    overall = agg.get("overall", {})
    rows = det.get("items", [])
    pass_n = overall.get("pass", 0)
    total_n = overall.get("total", 0)
    pass_rate = overall.get("rate", 0.0)
    token = (ragas.get("token_summary") or {}) if ragas else {}
    failures = report.get("top_failures", [])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    lines = [
        "PSA Passive Safety RAG — Evaluation Scorecard",
        "=" * 52,
        f"Generated: {report.get('generated_at', '')[:19]}",
        f"Golden set: {total_n} questions",
        "",
        "STAGE A — DETERMINISTIC (zero judge tokens)",
        f"  Overall pass: {pass_n}/{total_n} ({pass_rate:.1%})",
    ]
    for qt, stats in by_type.items():
        lines.append(f"  {qt}: {stats['pass']}/{stats['total']} ({stats['rate']:.1%})")
    lines.extend(["", "STAGE B — RAGAS"])
    if ar.get("mean") is not None:
        lines.append(f"  Answer relevancy (local): {ar['mean']:.3f}")
    if jm.get("faithfulness_mean") is not None:
        lines.append(f"  Faithfulness (judge):     {jm['faithfulness_mean']:.3f}")
    if jm.get("context_precision_mean") is not None:
        lines.append(f"  Context precision (judge): {jm['context_precision_mean']:.3f}")
    if token:
        lines.extend([
            "",
            "GROQ JUDGE USAGE",
            f"  Calls: {token.get('groq_calls', 0)}  |  Tokens: {token.get('total_tokens', 0)}"
            f" / {token.get('budget', 0)}",
        ])
    lines.extend(["", f"FAILURES ({len(failures)}):"])
    if failures:
        for f in failures[:8]:
            lines.append(f"  - {f['id']}: {', '.join(f.get('failures', []))}")
    else:
        lines.append("  (none)")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    score_path = out_dir / "eval_overall_scorecard.png"
    fig.savefig(score_path, dpi=150)
    plt.close(fig)
    written.append(score_path)

    return written
