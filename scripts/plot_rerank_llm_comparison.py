#!/usr/bin/env python3
"""Render benchmark chart from rerank_llm_comparison_5q.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "output" / "evaluation" / "current"
IN_JSON = EVAL_DIR / "rerank_llm_comparison_5q.json"
OUT_PNG = EVAL_DIR / "rerank_llm_comparison_5q.png"

METRICS = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Answer rel."),
    ("context_precision", "Ctx precision"),
    ("context_recall", "Ctx recall"),
    ("overall", "Overall"),
]
COLORS = ["#ea580c", "#2563eb", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]


def plot(data: dict, out: Path = OUT_PNG) -> Path:
    ranking = data.get("ranking") or []
    if not ranking:
        raise ValueError("No ranking data in JSON")

    labels = [f"{r['reranker']}\n+ {r['llm'].split()[0]} {r['llm'].split()[1] if len(r['llm'].split()) > 1 else ''}" for r in ranking]
    short_labels = [f"#{r['rank']}" for r in ranking]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.28)

    # Panel 1: overall ranking
    ax0 = fig.add_subplot(gs[0, :])
    overall = [r["overall"] for r in ranking]
    bars = ax0.barh(range(len(ranking)), overall, color=COLORS[: len(ranking)])
    ax0.set_yticks(range(len(ranking)))
    ax0.set_yticklabels(
        [f"#{r['rank']}  {r['reranker']} + {r['llm']}" for r in ranking],
        fontsize=10,
    )
    ax0.invert_yaxis()
    ax0.set_xlim(0, 1.05)
    ax0.set_xlabel("Overall RAGAS score (5 questions)")
    ax0.set_title("Reranker × LLM combinations — ranked by overall score", fontweight="bold")
    ax0.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, overall):
        ax0.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)

    best = data.get("best_combo", {})
    if best:
        ax0.text(
            0.02,
            0.02,
            f"Best: {best.get('reranker', '?').split('/')[-1]} + "
            f"{best.get('llm', '?')}  →  {best.get('overall_score', 0):.3f}",
            transform=ax0.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#fff7ed", edgecolor="#fed7aa"),
        )

    # Panel 2: metric breakdown (grouped bars)
    ax1 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(METRICS))
    w = 0.12
    for i, row in enumerate(ranking):
        vals = [row.get(k, 0) for k, _ in METRICS]
        offset = (i - len(ranking) / 2 + 0.5) * w
        ax1.bar(x + offset, vals, w, label=f"#{row['rank']}", color=COLORS[i % len(COLORS)])
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in METRICS], rotation=25, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Score")
    ax1.set_title("All metrics by combo", fontweight="bold")
    ax1.legend(fontsize=7, ncol=2, loc="lower right")
    ax1.grid(axis="y", alpha=0.3)

    # Panel 3: LLM split — avg overall per LLM across rerankers
    ax2 = fig.add_subplot(gs[1, 1])
    llm_groups: dict[str, list[float]] = {}
    for row in ranking:
        llm_groups.setdefault(row["llm"], []).append(row["overall"])
    llm_names = list(llm_groups.keys())
    llm_avgs = [float(np.mean(llm_groups[k])) for k in llm_names]
    rerank_groups: dict[str, list[float]] = {}
    for row in ranking:
        rerank_groups.setdefault(row["reranker"], []).append(row["overall"])
    rer_names = list(rerank_groups.keys())
    rer_avgs = [float(np.mean(rerank_groups[k])) for k in rer_names]

    x2 = np.arange(max(len(llm_names), len(rer_names)))
    ax2.bar(x2[: len(llm_names)] - 0.2, llm_avgs, 0.35, label="By LLM (avg)", color="#2563eb")
    ax2.bar(x2[: len(rer_names)] + 0.2, rer_avgs, 0.35, label="By reranker (avg)", color="#ea580c")
    tick_labels = []
    for i in range(max(len(llm_names), len(rer_names))):
        parts = []
        if i < len(llm_names):
            parts.append(llm_names[i].replace("Llama ", "L"))
        if i < len(rer_names):
            parts.append(rer_names[i].replace("bge-reranker-v2-m3", "bge-v2-m3"))
        tick_labels.append("\n".join(parts) if parts else "")
    ax2.set_xticks(x2[: max(len(llm_names), len(rer_names))])
    ax2.set_xticklabels(tick_labels[: max(len(llm_names), len(rer_names))], fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Avg overall")
    ax2.set_title("Average overall by LLM vs reranker", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    n = data.get("test_cases", 5)
    judge = data.get("ragas_judge", "?")
    ts = data.get("timestamp", "")
    fig.suptitle(
        f"RAG ablation: 3 rerankers × 2 Groq LLMs  (n={n}, judge={judge})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    if ts:
        fig.text(0.99, 0.01, ts, ha="right", fontsize=8, color="#78716c")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_JSON
    if not path.exists():
        print(f"Missing {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = plot(data)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
