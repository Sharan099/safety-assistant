#!/usr/bin/env python3
"""
5-question RAG ablation: 2 Groq LLMs × 3 rerankers.

Rerankers:
  - BAAI/bge-reranker-v2-m3   (current v3.2 default)
  - jinaai/jina-reranker-v3
  - Qwen/Qwen3-Reranker-0.6B

LLMs (Groq):
  - llama-3.1-8b-instant
  - llama-3.3-70b-versatile

Outputs: output/evaluation/current/rerank_llm_comparison_5q.json
"""

from __future__ import annotations

import gc
import json
import os
import statistics
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

from config import GROQ_MODEL, RAGAS_JUDGE_MAX_TOKENS
from backend.app.core.settings import TOP_K_AFTER_RERANK
from backend.app.llm.groq_client import GroqLLM
from backend.app.retrieval.hybrid import HybridRetriever

# Reuse eval helpers from the full suite
from tests.run_full_evaluation import (  # noqa: E402
    GroqRateLimitError,
    _build_prompt,
    _contexts_from_docs,
    _hybrid_rerank,
    _proxy_answer_relevancy,
    _proxy_context_precision,
    _proxy_context_recall,
    _proxy_faithfulness,
    p,
    run_ragas_metrics,
)

EVAL_DIR = ROOT / "output" / "evaluation" / "current"
TEST_CASES = ROOT / "tests" / "test_cases_5.json"
OUT_JSON = EVAL_DIR / "rerank_llm_comparison_5q.json"
CHECKPOINT = EVAL_DIR / "rerank_llm_comparison_5q_checkpoint.json"

LLMS = [
    ("llama-3.1-8b-instant", "Llama 3.1 8B Instant"),
    ("llama-3.3-70b-versatile", "Llama 3.3 70B"),
]

RERANKERS = [
    ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3", "crossencoder"),
    ("jinaai/jina-reranker-v3", "jina-reranker-v3", "jina"),
    ("Qwen/Qwen3-Reranker-0.6B", "qwen3-reranker-0.6b", "crossencoder_qwen"),
]

RAGAS_JUDGE = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.3-70b-versatile")


def _doc_text(doc: dict) -> str:
    return (
        f"{doc.get('heading_path') or doc.get('title', '')} "
        f"{doc.get('parent_context', '')[:200]} "
        f"{doc.get('text', '')[:800]}"
    ).strip()


class BenchmarkReranker:
    """Unified reranker wrapper for CrossEncoder, Qwen, and Jina-v3 APIs."""

    def __init__(self, model_id: str, kind: str) -> None:
        self.model_id = model_id
        self.kind = kind
        self._model = None
        self._available = True
        self._load_error = ""

    def _load(self) -> None:
        if self._model is not None or not self._available:
            return
        try:
            if self.kind == "jina":
                from transformers import AutoModel

                self._model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                )
                self._model.eval()
            elif self.kind == "crossencoder_qwen":
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_id, trust_remote_code=True)
                if getattr(self._model, "tokenizer", None) is not None:
                    tok = self._model.tokenizer
                    if tok.pad_token is None and tok.eos_token:
                        tok.pad_token = tok.eos_token
                    if hasattr(self._model, "model") and hasattr(self._model.model, "config"):
                        self._model.model.config.pad_token_id = tok.pad_token_id
            else:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_id)
            p(f"  Loaded reranker: {self.model_id}")
        except Exception as exc:
            self._available = False
            self._load_error = str(exc)
            p(f"  Reranker load failed ({self.model_id}): {exc}")

    def rerank(self, query: str, documents: list[dict]) -> dict[str, Any]:
        t0 = time.perf_counter()
        if not documents:
            return {"documents": [], "latency_ms": 0, "reranker_used": False}

        self._load()
        if not self._available or self._model is None:
            top = documents[:TOP_K_AFTER_RERANK]
            for d in top:
                d["rerank_score"] = d.get("rrf_score", d.get("score", 0))
            return {
                "documents": top,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "reranker_used": False,
                "error": self._load_error,
            }

        docs_copy = [dict(d) for d in documents]
        texts = [_doc_text(d) for d in docs_copy]

        try:
            if self.kind == "jina":
                ranked = self._model.rerank(query=query, documents=texts, top_n=TOP_K_AFTER_RERANK)
                for r in ranked:
                    idx = int(r.get("index", r.get("corpus_id", 0)))
                    score = float(
                        r.get("relevance_score", r.get("score", r.get("relevance", 0)))
                    )
                    if 0 <= idx < len(docs_copy):
                        docs_copy[idx]["rerank_score"] = score
            else:
                pairs = [(query, t) for t in texts]
                scores = self._model.predict(pairs, show_progress_bar=False)
                for doc, score in zip(docs_copy, scores):
                    doc["rerank_score"] = float(score)

            ranked_docs = sorted(
                docs_copy,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True,
            )[:TOP_K_AFTER_RERANK]
        except Exception as exc:
            p(f"  Rerank predict failed ({self.model_id}): {exc}")
            ranked_docs = docs_copy[:TOP_K_AFTER_RERANK]
            for d in ranked_docs:
                d["rerank_score"] = d.get("rrf_score", d.get("score", 0))

        return {
            "documents": ranked_docs,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "reranker_used": True,
        }

    def unload(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class GroqLLMOverride(GroqLLM):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model


def _run_ragas_with_judge(rows: list[dict]) -> dict | None:
    prev = os.environ.get("GROQ_MODEL")
    os.environ["GROQ_MODEL"] = RAGAS_JUDGE
    try:
        return run_ragas_metrics(rows)
    finally:
        if prev is None:
            os.environ.pop("GROQ_MODEL", None)
        else:
            os.environ["GROQ_MODEL"] = prev


def _metrics_from_ragas(ragas: dict | None, rows: list[dict], hyb_prec: list[float], hyb_recall: list[float]) -> dict[str, float]:
    proxies = {
        "faithfulness": round(
            statistics.mean([_proxy_faithfulness(r["answer"], r["contexts"]) for r in rows]), 4
        ),
        "answer_relevancy": round(
            statistics.mean([_proxy_answer_relevancy(r["answer"], r["ground_truth"]) for r in rows]), 4
        ),
        "context_precision": round(statistics.mean(hyb_prec) if hyb_prec else 0.0, 4),
        "context_recall": round(statistics.mean(hyb_recall) if hyb_recall else 0.0, 4),
    }
    key_map = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy", "answer_relevance"],
        "context_precision": ["context_precision"],
        "context_recall": ["context_recall"],
    }
    out: dict[str, float] = {}
    for key, aliases in key_map.items():
        v = None
        if ragas:
            for rk, rv in ragas.items():
                rk_s = str(rk).lower()
                if any(a in rk_s for a in aliases):
                    try:
                        v = float(rv)
                        break
                    except (TypeError, ValueError):
                        pass
        out[key] = round(v if v is not None else proxies[key], 4)
    out["overall"] = round(statistics.mean(list(out.values())), 4)
    return out


def run_combo(
    retriever: HybridRetriever,
    reranker: BenchmarkReranker,
    llm_model: str,
    cases: list[dict],
) -> dict[str, Any]:
    # Load reranker before LLM calls to avoid stacking model loads mid-pipeline
    reranker._load()
    llm = GroqLLMOverride(llm_model)
    ragas_rows: list[dict] = []
    hyb_prec: list[float] = []
    hyb_recall: list[float] = []
    latencies: list[float] = []

    for idx, case in enumerate(cases, 1):
        q, gt = case["question"], case["ground_truth"]
        p(f"    Q{idx}/5: {q[:60]}...")
        hyb_docs, hyb_ms = _hybrid_rerank(q, retriever, reranker)
        hyb_ctx = _contexts_from_docs(hyb_docs)
        hyb_recall.append(_proxy_context_recall(q, hyb_ctx, gt))
        hyb_prec.append(_proxy_context_precision(hyb_ctx, gt))
        latencies.append(hyb_ms)

        try:
            resp = llm.generate(_build_prompt(q, "\n\n".join(hyb_ctx)))
            answer = resp["answer"]
        except Exception as exc:
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                raise GroqRateLimitError(str(exc)) from exc
            p(f"    LLM error: {exc}")
            answer = "Information not found in regulations."

        ragas_rows.append(
            {
                "question": q,
                "answer": answer,
                "contexts": hyb_ctx,
                "ground_truth": gt,
            }
        )
        time.sleep(0.5)

    p("    Running RAGAS (5 questions)...")
    ragas = _run_ragas_with_judge(ragas_rows)
    metrics = _metrics_from_ragas(ragas, ragas_rows, hyb_prec, hyb_recall)
    return {
        "metrics": metrics,
        "ragas_raw": ragas,
        "avg_pipeline_ms": round(statistics.mean(latencies), 2),
        "per_question": [
            {
                "id": c.get("id"),
                "question": c["question"],
                "answer": r["answer"][:500],
                "context_recall_proxy": hyb_recall[i],
                "context_precision_proxy": hyb_prec[i],
            }
            for i, (c, r) in enumerate(zip(cases, ragas_rows))
        ],
    }


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        p("ERROR: GROQ_API_KEY required")
        sys.exit(1)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEST_CASES, encoding="utf-8") as f:
        cases = json.load(f)

    # Resume partial runs after a crash
    results: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if CHECKPOINT.exists():
        try:
            ck = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
            results = ck.get("detailed_results", [])
            done_ids = {r["combo_id"] for r in results}
            p(f"Resuming — {len(done_ids)} combos already done")
        except Exception:
            pass

    p(f"Reranker × LLM comparison — {len(cases)} questions")
    p(f"RAGAS judge: {RAGAS_JUDGE} (max_tokens={RAGAS_JUDGE_MAX_TOKENS})")

    retriever = HybridRetriever()
    p("Warming up retriever (loads Nomic once)...")
    retriever.retrieve("UN R14 seat belt anchorage strength test load")

    for llm_id, llm_label in LLMS:
        p(f"\n=== LLM: {llm_label} ({llm_id}) ===")
        for model_id, slug, kind in RERANKERS:
            combo_name = f"{slug}__{llm_id.replace('.', '-')}"
            if combo_name in done_ids:
                p(f"\n-- Skip (done): {combo_name} --")
                continue
            p(f"\n-- Combo: reranker={slug} + llm={llm_id} --")
            reranker = BenchmarkReranker(model_id, kind)
            t0 = time.time()
            try:
                payload = run_combo(retriever, reranker, llm_id, cases)
            except GroqRateLimitError as exc:
                p(f"FATAL rate limit at {combo_name}: {exc}")
                sys.exit(1)
            except Exception as exc:
                p(f"  Combo failed ({combo_name}): {exc}")
                import traceback
                traceback.print_exc()
                reranker.unload()
                continue
            elapsed = round(time.time() - t0, 1)
            reranker.unload()

            entry = {
                "combo_id": combo_name,
                "llm": llm_id,
                "llm_label": llm_label,
                "reranker": model_id,
                "reranker_slug": slug,
                "elapsed_s": elapsed,
                **payload,
            }
            results.append(entry)
            done_ids.add(combo_name)
            m = payload["metrics"]
            p(
                f"  → overall={m['overall']:.3f}  "
                f"faith={m['faithfulness']:.3f}  "
                f"ans_rel={m['answer_relevancy']:.3f}  "
                f"ctx_prec={m['context_precision']:.3f}  "
                f"ctx_rec={m['context_recall']:.3f}"
            )
            CHECKPOINT.write_text(
                json.dumps({"detailed_results": results}, indent=2),
                encoding="utf-8",
            )

    if not results:
        p("ERROR: no successful combinations")
        sys.exit(1)

    ranked = sorted(results, key=lambda x: x["metrics"]["overall"], reverse=True)
    best = ranked[0]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_cases": len(cases),
        "ragas_judge": RAGAS_JUDGE,
        "combinations_tested": len(results),
        "best_combo": {
            "combo_id": best["combo_id"],
            "llm": best["llm"],
            "reranker": best["reranker"],
            "overall_score": best["metrics"]["overall"],
            "metrics": best["metrics"],
        },
        "ranking": [
            {
                "rank": i + 1,
                "combo_id": r["combo_id"],
                "llm": r["llm_label"],
                "reranker": r["reranker_slug"],
                "overall": r["metrics"]["overall"],
                "faithfulness": r["metrics"]["faithfulness"],
                "answer_relevancy": r["metrics"]["answer_relevancy"],
                "context_precision": r["metrics"]["context_precision"],
                "context_recall": r["metrics"]["context_recall"],
                "avg_pipeline_ms": r["avg_pipeline_ms"],
            }
            for i, r in enumerate(ranked)
        ],
        "detailed_results": results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    p("\n" + "=" * 60)
    p("FINAL RANKING (overall RAGAS score, 5 questions)")
    p("=" * 60)
    for row in summary["ranking"]:
        p(
            f"  #{row['rank']} {row['reranker']} + {row['llm']} "
            f"→ overall {row['overall']:.3f}"
        )
    p(f"\nBest: {best['reranker_slug']} + {best['llm_label']} "
      f"(overall {best['metrics']['overall']:.3f})")
    p(f"Saved: {OUT_JSON}")

    try:
        import importlib.util

        plot_path = ROOT / "scripts" / "plot_rerank_llm_comparison.py"
        spec = importlib.util.spec_from_file_location("plot_rerank_llm", plot_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        png = mod.plot(summary)
        p(f"Chart: {png}")
    except Exception as exc:
        p(f"Plot failed (run scripts/plot_rerank_llm_comparison.py): {exc}")


if __name__ == "__main__":
    main()
