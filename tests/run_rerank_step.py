#!/usr/bin/env python3
"""
Run one RAG ablation step at a time (5 questions, RAGAS).

Steps (user workflow):
  1. llama-3.1-8b-instant  + jina-reranker-v3
  2. llama-3.1-8b-instant  + Qwen3-Reranker-0.6B
  3. llama-3.3-70b-versatile + jina-reranker-v3
  4. llama-3.3-70b-versatile + Qwen3-Reranker-0.6B

Usage:
  conda activate rag
  python tests/run_rerank_step.py --step 1
  python tests/run_rerank_step.py --finalize
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from config import RAGAS_JUDGE_MAX_TOKENS
from backend.app.retrieval.hybrid import HybridRetriever
from tests.run_rerank_llm_comparison import (  # noqa: E402
    BenchmarkReranker,
    RAGAS_JUDGE,
    run_combo,
)
from tests.run_full_evaluation import GroqRateLimitError, p  # noqa: E402

EVAL_DIR = ROOT / "output" / "evaluation" / "current"
STEPS_DIR = EVAL_DIR / "steps"
TEST_CASES = ROOT / "tests" / "test_cases_5.json"
FINAL_JSON = EVAL_DIR / "rerank_llm_comparison_5q.json"

STEPS: list[dict[str, Any]] = [
    {
        "step": 1,
        "label": "8B Instant + Jina-v3",
        "llm": "llama-3.1-8b-instant",
        "llm_label": "Llama 3.1 8B Instant",
        "reranker": "jinaai/jina-reranker-v3",
        "slug": "jina-reranker-v3",
        "kind": "jina",
    },
    {
        "step": 2,
        "label": "8B Instant + Qwen3-0.6B",
        "llm": "llama-3.1-8b-instant",
        "llm_label": "Llama 3.1 8B Instant",
        "reranker": "Qwen/Qwen3-Reranker-0.6B",
        "slug": "qwen3-reranker-0.6b",
        "kind": "crossencoder_qwen",
    },
    {
        "step": 3,
        "label": "70B + Jina-v3",
        "llm": "llama-3.3-70b-versatile",
        "llm_label": "Llama 3.3 70B",
        "reranker": "jinaai/jina-reranker-v3",
        "slug": "jina-reranker-v3",
        "kind": "jina",
    },
    {
        "step": 4,
        "label": "70B + Qwen3-0.6B",
        "llm": "llama-3.3-70b-versatile",
        "llm_label": "Llama 3.3 70B",
        "reranker": "Qwen/Qwen3-Reranker-0.6B",
        "slug": "qwen3-reranker-0.6b",
        "kind": "crossencoder_qwen",
    },
]


def step_path(step: int) -> Path:
    return STEPS_DIR / f"step{step}_5q.json"


def _combo_id(cfg: dict) -> str:
    return f"{cfg['slug']}__{cfg['llm'].replace('.', '-')}"


def import_checkpoint(step_cfg: dict) -> dict | None:
    ck_path = EVAL_DIR / "rerank_llm_comparison_5q_checkpoint.json"
    if not ck_path.exists():
        return None
    try:
        ck = json.loads(ck_path.read_text(encoding="utf-8"))
        cid = _combo_id(step_cfg)
        for row in ck.get("detailed_results", []):
            if row.get("combo_id") == cid:
                return row
    except Exception:
        pass
    return None


def run_step(step_num: int, *, force: bool = False) -> dict:
    cfg = next(s for s in STEPS if s["step"] == step_num)
    out = step_path(step_num)
    STEPS_DIR.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        p(f"Step {step_num} already saved → {out} (use --force to re-run)")
        return json.loads(out.read_text(encoding="utf-8"))

    imported = import_checkpoint(cfg)
    if imported and not force:
        entry = {**imported, "step": step_num, "step_label": cfg["label"]}
        out.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
        p(f"Step {step_num} imported from checkpoint → {out}")
        m = entry["metrics"]
        p(
            f"  overall={m['overall']:.3f} faith={m['faithfulness']:.3f} "
            f"ans_rel={m['answer_relevancy']:.3f} ctx_prec={m['context_precision']:.3f} "
            f"ctx_rec={m['context_recall']:.3f}"
        )
        return entry

    if not os.getenv("GROQ_API_KEY"):
        p("ERROR: GROQ_API_KEY required")
        sys.exit(1)

    with open(TEST_CASES, encoding="utf-8") as f:
        cases = json.load(f)

    p("=" * 60)
    p(f"STEP {step_num}/4: {cfg['label']}")
    p(f"  LLM      : {cfg['llm']}")
    p(f"  Reranker : {cfg['reranker']}")
    p(f"  Questions: {len(cases)}  |  RAGAS judge: {RAGAS_JUDGE}")
    p("=" * 60)

    retriever = HybridRetriever()
    p("Loading retriever (Nomic)...")
    retriever.retrieve("UN R14 seat belt anchorage strength")

    reranker = BenchmarkReranker(cfg["reranker"], cfg["kind"])
    t0 = time.time()
    try:
        payload = run_combo(retriever, reranker, cfg["llm"], cases)
    except GroqRateLimitError as exc:
        p(f"FATAL Groq rate limit: {exc}")
        sys.exit(1)
    finally:
        reranker.unload()
        gc.collect()

    entry = {
        "step": step_num,
        "step_label": cfg["label"],
        "combo_id": _combo_id(cfg),
        "llm": cfg["llm"],
        "llm_label": cfg["llm_label"],
        "reranker": cfg["reranker"],
        "reranker_slug": cfg["slug"],
        "elapsed_s": round(time.time() - t0, 1),
        **payload,
    }
    out.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    m = entry["metrics"]
    p(
        f"\nStep {step_num} done → overall={m['overall']:.3f} "
        f"(faith={m['faithfulness']:.3f}, ans_rel={m['answer_relevancy']:.3f}, "
        f"ctx_prec={m['context_precision']:.3f}, ctx_rec={m['context_recall']:.3f})"
    )
    p(f"Saved: {out}")
    return entry


def finalize() -> dict:
    results: list[dict] = []
    missing = []
    for cfg in STEPS:
        path = step_path(cfg["step"])
        if not path.exists():
            missing.append(cfg["step"])
            continue
        results.append(json.loads(path.read_text(encoding="utf-8")))

    if missing:
        p(f"ERROR: missing steps {missing}. Run: python tests/run_rerank_step.py --step N")
        sys.exit(1)

    ranked = sorted(results, key=lambda x: x["metrics"]["overall"], reverse=True)
    best = ranked[0]
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_cases": 5,
        "ragas_judge": RAGAS_JUDGE,
        "ragas_judge_max_tokens": RAGAS_JUDGE_MAX_TOKENS,
        "combinations_tested": len(results),
        "best_combo": {
            "step": best["step"],
            "step_label": best["step_label"],
            "combo_id": best["combo_id"],
            "llm": best["llm"],
            "reranker": best["reranker"],
            "overall_score": best["metrics"]["overall"],
            "metrics": best["metrics"],
        },
        "ranking": [
            {
                "rank": i + 1,
                "step": r["step"],
                "step_label": r["step_label"],
                "combo_id": r["combo_id"],
                "llm": r["llm_label"],
                "reranker": r["reranker_slug"],
                "overall": r["metrics"]["overall"],
                "faithfulness": r["metrics"]["faithfulness"],
                "answer_relevancy": r["metrics"]["answer_relevancy"],
                "context_precision": r["metrics"]["context_precision"],
                "context_recall": r["metrics"]["context_recall"],
                "avg_pipeline_ms": r["avg_pipeline_ms"],
                "elapsed_s": r.get("elapsed_s"),
            }
            for i, r in enumerate(ranked)
        ],
        "detailed_results": results,
    }
    FINAL_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    p("\n" + "=" * 60)
    p("FINAL COMPARISON (5 questions, RAGAS overall)")
    p("=" * 60)
    for row in summary["ranking"]:
        p(
            f"  #{row['rank']} Step {row['step']}: {row['step_label']} "
            f"→ overall {row['overall']:.3f}"
        )
    p(
        f"\n★ Best: Step {best['step']} — {best['step_label']} "
        f"(overall {best['metrics']['overall']:.3f})"
    )
    p(f"  LLM: {best['llm']}")
    p(f"  Reranker: {best['reranker']}")
    p(f"JSON: {FINAL_JSON}")

    plot_path = ROOT / "scripts" / "plot_rerank_llm_comparison.py"
    spec = importlib.util.spec_from_file_location("plot_rerank_llm", plot_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    png = mod.plot(summary)
    p(f"Chart: {png}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Step-by-step reranker×LLM RAG eval (5Q)")
    ap.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Run a single step")
    ap.add_argument("--finalize", action="store_true", help="Merge steps + plot + pick best")
    ap.add_argument("--force", action="store_true", help="Re-run even if step file exists")
    args = ap.parse_args()

    if args.finalize:
        finalize()
        return
    if args.step is None:
        ap.print_help()
        sys.exit(1)
    run_step(args.step, force=args.force)


if __name__ == "__main__":
    main()
