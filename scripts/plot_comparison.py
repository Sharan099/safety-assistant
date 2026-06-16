#!/usr/bin/env python3
"""
Render side-by-side comparison plots from evaluation JSON artifacts.

Produces two multi-panel figures under output/evaluation/:
  1. eval_compare_reranker_5q_groq.png
       Nomic + bge-reranker-v2-m3  vs  Nomic + bge-reranker-base
       (5 questions, live Groq generation, RAGAS "full" mode)
  2. eval_compare_embedding_20q.png
       BGE-base + bge-reranker-base  vs  Nomic + bge-reranker-v2-m3
       (20 questions, retrieval-proxy mode)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "output" / "evaluation"

RAGAS_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
RAGAS_LABELS = ["Faithful.", "Ans. rel.", "Ctx prec.", "Ctx recall"]
LAT_KEYS = ["semantic_avg_ms", "hybrid_avg_ms", "pipeline_p95_ms"]
LAT_LABELS = ["Semantic\navg", "Hybrid\navg", "Pipeline\np95"]

COLOR_A = "#2563eb"  # blue
COLOR_B = "#f59e0b"  # amber


def load(name: str) -> dict:
    with open(EVAL_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _short(models: dict) -> str:
    emb = models.get("embedding", "?").split("/")[-1]
    rer = models.get("reranker", "?").split("/")[-1]
    return f"{emb} + {rer}"


def _ragas_panel(ax, a: dict, b: dict, label_a: str, label_b: str, title: str) -> None:
    av = [a.get("ragas", {}).get(k, 0.0) for k in RAGAS_KEYS] + [a.get("overall_score", 0.0)]
    bv = [b.get("ragas", {}).get(k, 0.0) for k in RAGAS_KEYS] + [b.get("overall_score", 0.0)]
    labels = RAGAS_LABELS + ["Overall"]
    x = range(len(labels))
    w = 0.38
    bars_a = ax.bar([i - w / 2 for i in x], av, w, label=label_a, color=COLOR_A)
    bars_b = ax.bar([i + w / 2 for i in x], bv, w, label=label_b, color=COLOR_B)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0-1)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)


def _latency_panel(ax, a: dict, b: dict, label_a: str, label_b: str, title: str) -> None:
    av = [a.get("latency", {}).get(k, 0.0) / 1000.0 for k in LAT_KEYS]
    bv = [b.get("latency", {}).get(k, 0.0) / 1000.0 for k in LAT_KEYS]
    x = range(len(LAT_LABELS))
    w = 0.38
    bars_a = ax.bar([i - w / 2 for i in x], av, w, label=label_a, color=COLOR_A)
    bars_b = ax.bar([i + w / 2 for i in x], bv, w, label=label_b, color=COLOR_B)
    ax.set_xticks(list(x))
    ax.set_xticklabels(LAT_LABELS, fontsize=9)
    ax.set_ylabel("Seconds (lower is better)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=7)


def figure(file_a: str, file_b: str, out_name: str, suptitle: str) -> Path:
    a, b = load(file_a), load(file_b)
    label_a, label_b = _short(a["models"]), _short(b["models"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _ragas_panel(ax1, a, b, label_a, label_b, "RAGAS quality")
    _latency_panel(ax2, a, b, label_a, label_b, "Latency")
    n = a.get("test_cases_total", "?")
    mode = a.get("evaluation_mode", "?").split(" ")[0]
    fig.suptitle(f"{suptitle}  (n={n} questions, mode={mode})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = EVAL_DIR / out_name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    outputs = [
        figure(
            "rag_eval_5_groq_nomic_v2m3.json",
            "rag_eval_5_groq_nomic_baseRerank.json",
            "eval_compare_reranker_5q_groq.png",
            "Reranker comparison (Nomic embeddings, live Groq)",
        ),
        figure(
            "rag_eval_20_results.bge_baseline.json",
            "rag_eval_20_results.nomic_v2m3.json",
            "eval_compare_embedding_20q.png",
            "Embedding+reranker upgrade (BGE baseline vs Nomic+v2-m3)",
        ),
    ]
    for o in outputs:
        print(f"Saved {o}")


if __name__ == "__main__":
    main()
