"""
RAGAS evaluation for AutoSafety RAG.
Compares baseline (semantic-only) vs full hybrid + rerank pipeline.
Saves metrics JSON and comparison PNG under output/.
"""

import json
import os

# Windows OpenMP fix — must precede torch/sentence-transformers import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from backend.app.core.settings import EMBEDDING_MODEL, GROQ_MODEL, RERANKER_MODEL
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker

OUTPUT_DIR = ROOT / "output"
TEST_CASES = ROOT / "tests" / "test_cases.json"


def _semantic_only(query: str, retriever: HybridRetriever) -> list[dict]:
    regs = retriever._detect_regs(query)
    allowed_ids = retriever._filter_chunk_ids(regs)
    return retriever._semantic_search(query, allowed_ids)


def _full_pipeline(query: str, retriever: HybridRetriever, reranker: CrossEncoderReranker) -> list[dict]:
    r = retriever.retrieve(query)
    out = reranker.rerank(query, r["documents"])
    return out["documents"]


def _context_from_docs(docs: list[dict]) -> str:
    return "\n".join(
        f"{d.get('title', '')}: {(d.get('text', '') or '')[:500]}"
        for d in docs[:5]
    )


def _simple_scores(question: str, contexts: list[str], answer: str, ground_truth: str) -> dict:
    """Lightweight metrics when RAGAS/Groq unavailable (offline-friendly)."""
    ctx = " ".join(contexts).lower()
    q_words = {w for w in question.lower().split() if len(w) > 4}
    gt_words = {w for w in ground_truth.lower().split() if len(w) > 4}
    ctx_hits = sum(1 for w in q_words if w in ctx) / max(len(q_words), 1)
    faithfulness = sum(1 for w in gt_words if w in (answer or "").lower()) / max(
        len(gt_words), 1
    )
    return {
        "context_recall_proxy": round(ctx_hits, 4),
        "answer_relevance_proxy": round(faithfulness, 4),
    }


def run_ragas_if_available(dataset_rows: list[dict]) -> dict | None:
    if not os.getenv("GROQ_API_KEY"):
        return None
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        ds = Dataset.from_list(dataset_rows)
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        return dict(result)
    except Exception as exc:
        print(f"RAGAS skipped: {exc}")
        return None


def main() -> None:
    with open(TEST_CASES, encoding="utf-8") as f:
        cases = json.load(f)

    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    baseline_rows = []
    hybrid_rows = []
    ragas_dataset = []

    print("Running evaluation on", len(cases), "test cases…")

    for case in cases:
        q = case["question"]
        gt = case["ground_truth"]

        t0 = time.perf_counter()
        sem_docs = _semantic_only(q, retriever)
        baseline_latency = (time.perf_counter() - t0) * 1000
        baseline_ctx = [_context_from_docs(sem_docs)]

        t0 = time.perf_counter()
        hyb_docs = _full_pipeline(q, retriever, reranker)
        hybrid_latency = (time.perf_counter() - t0) * 1000
        hybrid_ctx = [_context_from_docs(hyb_docs)]

        answer = f"Retrieved context covers: {hyb_docs[0].get('title', 'N/A') if hyb_docs else 'none'}"

        b_scores = _simple_scores(q, baseline_ctx, answer, gt)
        h_scores = _simple_scores(q, hybrid_ctx, answer, gt)

        baseline_rows.append(
            {
                "question": q,
                "latency_ms": round(baseline_latency, 2),
                **b_scores,
            }
        )
        hybrid_rows.append(
            {
                "question": q,
                "latency_ms": round(hybrid_latency, 2),
                **h_scores,
            }
        )
        ragas_dataset.append(
            {
                "question": q,
                "answer": answer,
                "contexts": hybrid_ctx,
                "ground_truth": gt,
            }
        )

    ragas_scores = run_ragas_if_available(ragas_dataset)

    def avg(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "test_cases": len(cases),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_semantic_only": {
            "avg_context_recall_proxy": avg(baseline_rows, "context_recall_proxy"),
            "avg_answer_relevance_proxy": avg(baseline_rows, "answer_relevance_proxy"),
            "avg_latency_ms": avg(baseline_rows, "latency_ms"),
            "per_case": baseline_rows,
        },
        "hybrid_rrf_rerank": {
            "avg_context_recall_proxy": avg(hybrid_rows, "context_recall_proxy"),
            "avg_answer_relevance_proxy": avg(hybrid_rows, "answer_relevance_proxy"),
            "avg_latency_ms": avg(hybrid_rows, "latency_ms"),
            "per_case": hybrid_rows,
        },
        "ragas": ragas_scores,
        "models": {
            "embedding": EMBEDDING_MODEL,
            "llm": GROQ_MODEL,
            "reranker": RERANKER_MODEL,
        },
    }

    out_json = OUTPUT_DIR / "rag_evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")

    labels = ["Context Recall", "Answer Relevance", "Latency (ms)"]
    baseline_vals = [
        summary["baseline_semantic_only"]["avg_context_recall_proxy"],
        summary["baseline_semantic_only"]["avg_answer_relevance_proxy"],
        summary["baseline_semantic_only"]["avg_latency_ms"] / 1000,
    ]
    hybrid_vals = [
        summary["hybrid_rrf_rerank"]["avg_context_recall_proxy"],
        summary["hybrid_rrf_rerank"]["avg_answer_relevance_proxy"],
        summary["hybrid_rrf_rerank"]["avg_latency_ms"] / 1000,
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_vals[:2] + [0], width, label="Semantic only", color="#94a3b8")
    ax.bar(x + width / 2, hybrid_vals[:2] + [0], width, label="Hybrid + RRF + Rerank", color="#3b82f6")
    ax2 = ax.twinx()
    ax2.bar(
        x[-1] - width / 2,
        baseline_vals[2],
        width,
        color="#94a3b8",
        alpha=0.5,
    )
    ax2.bar(
        x[-1] + width / 2,
        hybrid_vals[2],
        width,
        color="#3b82f6",
        alpha=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (0–1 for metrics)")
    ax2.set_ylabel("Latency (seconds)")
    ax.set_title("RAG Evaluation: Baseline vs Hybrid + Rerank")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    png_path = OUTPUT_DIR / "rag_evaluation_comparison.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {png_path}")

    print("\n=== Summary ===")
    print(
        "Baseline context recall:",
        summary["baseline_semantic_only"]["avg_context_recall_proxy"],
    )
    print(
        "Hybrid context recall:",
        summary["hybrid_rrf_rerank"]["avg_context_recall_proxy"],
    )


if __name__ == "__main__":
    main()
