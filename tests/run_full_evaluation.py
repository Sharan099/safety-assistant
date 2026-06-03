#!/usr/bin/env python3
"""
Full RAG evaluation (70 questions):
  - RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
  - Ablation: semantic-only vs hybrid (BM25 + RRF + BGE rerank)
  - Corpus scale, guardrails effect, latency/cost
  - PNG dashboards under output/evaluation/
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from config import CHUNKS_FILE, DATA_DIR, EMBEDDINGS_FILE, GROQ_MODEL, MARKDOWN_DIR
from backend.app.core.settings import EMBEDDING_MODEL, RERANKER_MODEL
from backend.app.guardrails.validator import SafetyGuardrails
from backend.app.llm.groq_client import GroqLLM
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker

EVAL_DIR = ROOT / "output" / "evaluation"
TEST_CASES = Path(os.getenv("EVAL_TEST_CASES", str(ROOT / "tests" / "test_cases_70.json")))
CHECKPOINT = EVAL_DIR / "eval_checkpoint.json"
RESULTS_JSON = EVAL_DIR / os.getenv("EVAL_RESULTS_NAME", "rag_full_evaluation_results.json")
SKIP_LLM = os.getenv("EVAL_SKIP_LLM", "false").lower() == "true"

NOT_FOUND_PHRASES = (
    "information not found",
    "not found in regulations",
    "not found in the regulation",
    "cannot find",
    "no information",
)


def p(msg: str) -> None:
    print(msg, flush=True)


def ensure_test_cases() -> None:
    if not TEST_CASES.exists():
        from tests.generate_test_cases_70 import main as gen

        gen()


def corpus_stats() -> dict:
    pdfs = sorted(DATA_DIR.rglob("*.pdf"))
    pdfs = [x for x in pdfs if x.is_file()]
    chunks_n = 0
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            data = json.load(f)
            chunks_n = data.get("total_chunks") or len(data.get("chunks", []))

    pages = 0
    md_files = list(MARKDOWN_DIR.glob("*.md")) if MARKDOWN_DIR.exists() else []
    for md in md_files:
        text = md.read_text(encoding="utf-8", errors="ignore")
        pages += len(re.findall(r"^## Page \d+", text, re.MULTILINE))

    embeddings_n = 0
    emb_model = None
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, encoding="utf-8") as f:
            ed = json.load(f)
            embeddings_n = ed.get("total_vectors", len(ed.get("embeddings", {})))
            emb_model = ed.get("model")

    regs_indexed = set()
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            for c in json.load(f).get("chunks", []):
                if c.get("regulation"):
                    regs_indexed.add(c["regulation"])

    return {
        "regulation_pdfs_total": len(pdfs),
        "regulation_pdfs_indexed": len(md_files),
        "total_pages_ocr_markdown": pages,
        "chunks_indexed": chunks_n,
        "embeddings_indexed": embeddings_n,
        "embedding_model_artifact": emb_model or EMBEDDING_MODEL,
        "regulations_in_index": sorted(regs_indexed),
    }


def _contexts_from_docs(docs: list[dict], k: int = 5) -> list[str]:
    out = []
    for d in docs[:k]:
        title = d.get("heading_path") or d.get("title", "")
        text = (d.get("text") or "")[:1200]
        out.append(f"{title}\n{text}".strip())
    return out if out else [""]


def _build_prompt(query: str, context: str) -> str:
    return f"""You are PSA AI, a passive safety regulation assistant.

USER QUESTION
{query}

RETRIEVED REGULATION CONTEXT
{context}

Answer using ONLY the context above. Use clear markdown.
If information is missing, say: Information not found in regulations.
"""


def _answer_out_of_scope(answer: str) -> bool:
    low = (answer or "").lower()
    return any(p in low for p in NOT_FOUND_PHRASES)


def _hallucination_proxy(answer: str, contexts: list[str]) -> bool:
    """True if answer likely hallucinated (no 'not found' but weak context overlap)."""
    if _answer_out_of_scope(answer):
        return False
    ctx = " ".join(contexts).lower()
    if len(ctx.strip()) < 50:
        return True
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", answer.lower()).split() if len(w) > 5]
    if not words:
        return True
    hits = sum(1 for w in words[:20] if w in ctx)
    return hits / len(words[:20]) < 0.15


def _semantic_only(query: str, retriever: HybridRetriever) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    regs = retriever._detect_regs(query)
    allowed = retriever._filter_chunk_ids(regs)
    docs = retriever._semantic_search(query, allowed)
    return docs, (time.perf_counter() - t0) * 1000


def _hybrid_rerank(query: str, retriever: HybridRetriever, reranker: CrossEncoderReranker) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    r = retriever.retrieve(query)
    out = reranker.rerank(query, r["documents"])
    return out["documents"], (time.perf_counter() - t0) * 1000


def _proxy_context_recall(question: str, contexts: list[str], ground_truth: str) -> float:
    blob = " ".join(contexts).lower()
    q_terms = {w for w in question.lower().split() if len(w) > 4}
    gt_terms = {w for w in ground_truth.lower().split() if len(w) > 4}
    q_hit = sum(1 for w in q_terms if w in blob) / max(len(q_terms), 1)
    gt_hit = sum(1 for w in gt_terms if w in blob) / max(len(gt_terms), 1)
    return round(0.5 * q_hit + 0.5 * gt_hit, 4)


def _proxy_context_precision(contexts: list[str], ground_truth: str) -> float:
    if not contexts or not contexts[0].strip():
        return 0.0
    gt_terms = {w for w in ground_truth.lower().split() if len(w) > 4}
    scores = []
    for ctx in contexts[:3]:
        c = ctx.lower()
        scores.append(sum(1 for w in gt_terms if w in c) / max(len(gt_terms), 1))
    return round(float(np.mean(scores)), 4)


def _proxy_faithfulness(answer: str, contexts: list[str]) -> float:
    if _answer_out_of_scope(answer):
        return 1.0
    ctx = " ".join(contexts).lower()
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", answer.lower()).split() if len(w) > 4]
    if not words:
        return 0.0
    return round(sum(1 for w in words if w in ctx) / len(words), 4)


def _proxy_answer_relevancy(answer: str, ground_truth: str) -> float:
    gt = {w for w in ground_truth.lower().split() if len(w) > 4}
    ans = (answer or "").lower()
    if not gt:
        return 0.0
    return round(sum(1 for w in gt if w in ans) / len(gt), 4)


def _shim_langchain_vertexai() -> None:
    """ragas hard-imports langchain_community Vertex AI classes that were removed
    in newer langchain-community. We never use Vertex, so register harmless stubs
    so the (unused) imports resolve."""
    import sys
    import types

    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
        vertexai_chat_ok = True
    except Exception:
        vertexai_chat_ok = False

    if not vertexai_chat_ok:
        mod = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # minimal stub, never instantiated in our eval
            pass

        mod.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = mod
        try:
            import langchain_community.chat_models as _cm
            _cm.vertexai = mod
        except Exception:
            pass

    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except Exception:
        try:
            import langchain_community.llms as _llms

            class VertexAI:  # minimal stub
                pass

            _llms.VertexAI = VertexAI
        except Exception:
            pass


def run_ragas_metrics(rows: list[dict]) -> dict | None:
    if not os.getenv("GROQ_API_KEY"):
        p("RAGAS: skipped (no GROQ_API_KEY)")
        return None
    try:
        _shim_langchain_vertexai()
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import (
            answer_relevancy, context_precision, context_recall, faithfulness,
        )

        # Groq only supports n=1; default strictness=3 sends n=3 and gets a 400.
        answer_relevancy.strictness = 1

        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings

        # Judge LLM = your Groq model (bigger max_tokens so judge JSON isn't truncated)
        evaluator_llm = LangchainLLMWrapper(
            ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=2048)
        )
        # Embeddings = local + free. REQUIRED by answer_relevancy & context_precision.
        evaluator_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )

        ds = Dataset.from_list([
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in rows
        ])

        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_emb,                       # <- fixes Bug 2
            run_config=RunConfig(max_workers=1, timeout=180, max_retries=5),  # <- fixes Bug 3
        )
        # ragas >=0.2 exposes aggregated means via _repr_dict; older versions
        # support dict(result). Fall back to per-sample means if neither exists.
        agg = getattr(result, "_repr_dict", None)
        if agg is None:
            try:
                agg = dict(result)
            except Exception:
                scores = getattr(result, "_scores_dict", {}) or {}
                agg = {
                    k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()
                }
        out = {}
        for k, v in agg.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        return out
    except Exception as exc:
        import traceback
        p(f"RAGAS evaluate failed: {exc}")
        traceback.print_exc()   # <- fixes Bug 1: shows the REAL error instead of hiding it
        return None


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def _estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * 0.59 + completion_tokens * 0.79) / 1_000_000


def plot_ablation(summary: dict, path: Path) -> None:
    sem = summary["ablation"]["semantic_only"]
    hyb = summary["ablation"]["hybrid_rrf_rerank"]
    labels = ["Context Recall", "Context Precision"]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - w / 2,
        [sem["avg_context_recall_proxy"], sem["avg_context_precision_proxy"]],
        w,
        label="Semantic only",
        color="#94a3b8",
    )
    ax.bar(
        x + w / 2,
        [hyb["avg_context_recall_proxy"], hyb["avg_context_precision_proxy"]],
        w,
        label="Hybrid + RRF + Rerank",
        color="#3b82f6",
    )
    lift = summary["ablation"]["context_recall_lift_pct"]
    ax.set_title(f"Ablation: Hybrid vs Semantic-only (context recall +{lift:.1f}%)")
    ax.set_ylabel("Score (0–1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ragas_bars(metrics: dict, proxies: dict, path: Path) -> None:
    names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
    vals = [float(metrics.get(n, proxies.get(n, 0))) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#22c55e", "#3b82f6", "#8b5cf6", "#f59e0b"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("RAGAS Evaluation Metrics (Hybrid + RRF + Rerank)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overall_scorecard(summary: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    c = summary["corpus"]
    r = summary.get("ragas") or summary["ragas_proxies"]
    if summary.get("ragas"):
        r = summary["ragas"]
    else:
        r = {k: summary["ragas_proxies"].get(k, 0) for k in
             ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}
    ab = summary["ablation"]
    g = summary["guardrails"]
    lat = summary["latency"]
    overall = summary["overall_score"]

    lines = [
        "AutoSafety RAG — Evaluation Scorecard",
        "=" * 52,
        f"Test set: {summary['test_cases_regulation']} regulation + {summary['test_cases_guardrail']} guardrail = {summary['test_cases_total']} questions",
        "",
        "CORPUS SCALE",
        f"  PDFs in corpus: {c['regulation_pdfs_total']}  |  indexed (markdown): {c['regulation_pdfs_indexed']}",
        f"  Pages (OCR markdown): {c['total_pages_ocr_markdown']}  |  Chunks: {c['chunks_indexed']}  |  Vectors: {c['embeddings_indexed']}",
        f"  Regulations: {', '.join(c['regulations_in_index'])}",
        "",
        "RAGAS METRICS (hybrid pipeline)",
        f"  Faithfulness:        {float(r.get('faithfulness', 0)):.3f}",
        f"  Answer relevancy:    {float(r.get('answer_relevancy', 0)):.3f}",
        f"  Context precision:   {float(r.get('context_precision', 0)):.3f}",
        f"  Context recall:      {float(r.get('context_recall', 0)):.3f}",
        "",
        "ABLATION (retrieval)",
        f"  Semantic-only context recall:  {ab['semantic_only']['avg_context_recall_proxy']:.3f}",
        f"  Hybrid+RRF+Rerank recall:      {ab['hybrid_rrf_rerank']['avg_context_recall_proxy']:.3f}",
        f"  Improvement: +{ab['context_recall_lift_pct']:.1f}% context recall vs semantic-only",
        "",
        "GUARDRAILS (measured on test set)",
        f"  Input blocked (injection/jailbreak): {g['input_block_rate_pct']:.1f}%",
        f"  Out-of-scope safe answers:           {g['out_of_scope_safe_rate_pct']:.1f}%",
        f"  Hallucination proxy (regulation set): {g['hallucination_rate_semantic_pct']:.1f}% → {g['hallucination_rate_hybrid_pct']:.1f}%",
        "",
        "LATENCY / COST (observability-style)",
        f"  Retrieval p95: {lat['retrieval_p95_ms']:.0f} ms  |  Pipeline p95: {lat['pipeline_p95_ms']:.0f} ms",
        f"  Avg tokens/query: {lat['avg_prompt_tokens']:.0f} prompt + {lat['avg_completion_tokens']:.0f} completion",
        f"  Est. cost/query: ${lat['est_cost_per_query_usd']:.5f}",
        "",
        f"OVERALL SCORE: {overall:.3f} / 1.000",
        f"Models: embed={summary['models']['embedding']}, rerank={summary['models']['reranker']}, llm={summary['models']['llm']}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#cbd5e1"))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_guardrails(g: dict, path: Path) -> None:
    labels = ["Input blocked\n(injection)", "Out-of-scope\nsafe reply", "Hallucination\n(semantic)", "Hallucination\n(hybrid)"]
    vals = [
        g["input_block_rate_pct"] / 100,
        g["out_of_scope_safe_rate_pct"] / 100,
        g["hallucination_rate_semantic_pct"] / 100,
        g["hallucination_rate_hybrid_pct"] / 100,
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#22c55e", "#3b82f6", "#f97316", "#16a34a"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate (0–1)")
    ax.set_title("Guardrails & Hallucination Proxy (70-question test set)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v*100:.0f}%", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_latency(rows: list[dict], path: Path) -> None:
    sem = [r["semantic_retrieval_ms"] for r in rows]
    hyb = [r["hybrid_pipeline_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(sem, bins=20, alpha=0.6, label="Semantic retrieval", color="#94a3b8")
    ax.hist(hyb, bins=20, alpha=0.6, label="Hybrid+rerank pipeline", color="#3b82f6")
    ax.axvline(_p95(sem), color="#64748b", linestyle="--", label=f"Semantic p95={_p95(sem):.0f}ms")
    ax.axvline(_p95(hyb), color="#1d4ed8", linestyle="--", label=f"Hybrid p95={_p95(hyb):.0f}ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Retrieval / Pipeline Latency Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_test_cases()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEST_CASES, encoding="utf-8") as f:
        all_cases = json.load(f)

    reg_cases = [c for c in all_cases if c.get("category", "regulation") == "regulation"]
    guard_cases = [c for c in all_cases if c.get("category", "").startswith("guardrail")]

    p(f"Evaluation: {len(reg_cases)} regulation + {len(guard_cases)} guardrail questions")

    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()
    guardrails = SafetyGuardrails()
    llm: GroqLLM | None = None
    llm_disabled_reason = ""
    if SKIP_LLM:
        llm_disabled_reason = "EVAL_SKIP_LLM=true"
    elif os.getenv("GROQ_API_KEY"):
        try:
            llm = GroqLLM()
        except Exception as exc:
            llm_disabled_reason = str(exc)
            p(f"Groq LLM unavailable: {exc}")
    else:
        llm_disabled_reason = "GROQ_API_KEY not set"

    def _generate_answer(q: str, ctx: list[str]) -> tuple[str, int, int]:
        nonlocal llm, llm_disabled_reason
        if llm is None:
            top = (ctx[0] if ctx else "")[:400]
            return (
                f"Based on retrieved regulations: {top[:350]}..."
                if top.strip()
                else "Information not found in regulations.",
                0,
                0,
            )
        try:
            resp = llm.generate(_build_prompt(q, "\n\n".join(ctx)))
            return (
                resp["answer"],
                resp.get("prompt_tokens") or 500,
                resp.get("completion_tokens") or max(80, int(len(resp["answer"].split()) * 1.33)),
            )
        except Exception as exc:
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                llm = None
                llm_disabled_reason = "Groq rate limit — using retrieval proxy answers"
                p(f"  {llm_disabled_reason}")
            else:
                p(f"  LLM error: {exc}")
            top = (ctx[0] if ctx else "")[:400]
            return (
                "Information not found in regulations."
                if not top.strip()
                else f"Based on retrieved regulations: {top[:350]}...",
                0,
                0,
            )

    per_reg: list[dict] = []
    ragas_rows: list[dict] = []

    sem_recall, hyb_recall = [], []
    sem_prec, hyb_prec = [], []
    sem_lat, hyb_lat = [], []
    prompt_toks, completion_toks = [], []

    for idx, case in enumerate(reg_cases, 1):
        q, gt = case["question"], case["ground_truth"]
        p(f"[{idx}/{len(reg_cases)}] {q[:70]}...")

        sem_docs, sem_ms = _semantic_only(q, retriever)
        hyb_docs, hyb_ms = _hybrid_rerank(q, retriever, reranker)
        sem_ctx = _contexts_from_docs(sem_docs)
        hyb_ctx = _contexts_from_docs(hyb_docs)

        cr_sem = _proxy_context_recall(q, sem_ctx, gt)
        cr_hyb = _proxy_context_recall(q, hyb_ctx, gt)
        cp_sem = _proxy_context_precision(sem_ctx, gt)
        cp_hyb = _proxy_context_precision(hyb_ctx, gt)
        sem_recall.append(cr_sem)
        hyb_recall.append(cr_hyb)
        sem_prec.append(cp_sem)
        hyb_prec.append(cp_hyb)
        sem_lat.append(sem_ms)
        hyb_lat.append(hyb_ms)

        answer, pt, ct = _generate_answer(q, hyb_ctx)
        if pt:
            prompt_toks.append(pt)
            completion_toks.append(ct)
        faith = _proxy_faithfulness(answer, hyb_ctx)
        rel = _proxy_answer_relevancy(answer, gt)

        ragas_rows.append({
            "question": q,
            "answer": answer,
            "contexts": hyb_ctx,
            "ground_truth": gt,
        })

        ans_sem, _, _ = _generate_answer(q, sem_ctx)

        per_reg.append({
            "id": case.get("id"),
            "question": q,
            "semantic_retrieval_ms": round(sem_ms, 2),
            "hybrid_pipeline_ms": round(hyb_ms, 2),
            "context_recall_semantic": cr_sem,
            "context_recall_hybrid": cr_hyb,
            "context_precision_semantic": cp_sem,
            "context_precision_hybrid": cp_hyb,
            "faithfulness_proxy": faith,
            "answer_relevancy_proxy": rel,
            "hallucination_semantic": _hallucination_proxy(ans_sem, sem_ctx),
            "hallucination_hybrid": _hallucination_proxy(answer, hyb_ctx),
        })

    # Guardrails
    inj_blocked = 0
    inj_total = 0
    oos_safe = 0
    oos_total = 0
    for case in guard_cases:
        q = case["question"]
        cat = case.get("category", "")
        gr = guardrails.validate_input(q)
        if "injection" in cat or "jailbreak" in cat or "unsafe" in cat:
            inj_total += 1
            if gr.blocked:
                inj_blocked += 1
        if "out_of_scope" in cat:
            oos_total += 1
            hyb_docs, _ = _hybrid_rerank(q, retriever, reranker)
            ctx = _contexts_from_docs(hyb_docs)
            ans, _, _ = _generate_answer(q, ctx)
            # Safe if explicit not-found or weak regulation context for off-topic query
            if _answer_out_of_scope(ans) or _proxy_context_recall(q, ctx, "") < 0.1:
                oos_safe += 1

    # RAGAS on subset if full 60 too slow — user asked 70; run RAGAS on all reg with Groq
    p("Running RAGAS metrics (may take several minutes)...")
    ragas = run_ragas_metrics(ragas_rows)

    def avg(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    proxies = {
        "faithfulness": round(
            avg([_proxy_faithfulness(r["answer"], r["contexts"]) for r in ragas_rows]), 4
        ),
        "answer_relevancy": round(
            avg([_proxy_answer_relevancy(r["answer"], r["ground_truth"]) for r in ragas_rows]), 4
        ),
        "context_precision": round(avg(hyb_prec), 4),
        "context_recall": round(avg(hyb_recall), 4),
    }

    metrics_final = {}
    ragas_key_map = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy", "answer_relevance"],
        "context_precision": ["context_precision"],
        "context_recall": ["context_recall"],
    }
    for key, aliases in ragas_key_map.items():
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
        metrics_final[key] = round(v if v is not None else proxies[key], 4)

    sem_cr = avg(sem_recall)
    hyb_cr = avg(hyb_recall)
    lift_pct = ((hyb_cr - sem_cr) / sem_cr * 100) if sem_cr > 0 else 0.0

    hall_sem = sum(1 for r in per_reg if r.get("hallucination_semantic")) / max(len(per_reg), 1) * 100
    hall_hyb = sum(1 for r in per_reg if r.get("hallucination_hybrid")) / max(len(per_reg), 1) * 100

    overall = statistics.mean(list(metrics_final.values()))

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_mode": "full" if llm and not llm_disabled_reason else f"retrieval_proxy ({llm_disabled_reason or 'no LLM'})",
        "test_cases_total": len(all_cases),
        "test_cases_regulation": len(reg_cases),
        "test_cases_guardrail": len(guard_cases),
        "corpus": corpus_stats(),
        "ragas": metrics_final if ragas else None,
        "ragas_proxies": {**proxies, **{k: round(v, 4) for k, v in proxies.items()}},
        "ragas_raw": ragas,
        "ablation": {
            "semantic_only": {
                "avg_context_recall_proxy": round(sem_cr, 4),
                "avg_context_precision_proxy": round(avg(sem_prec), 4),
                "avg_latency_ms": round(avg(sem_lat), 2),
            },
            "hybrid_rrf_rerank": {
                "avg_context_recall_proxy": round(hyb_cr, 4),
                "avg_context_precision_proxy": round(avg(hyb_prec), 4),
                "avg_latency_ms": round(avg(hyb_lat), 2),
            },
            "context_recall_lift_pct": round(lift_pct, 2),
        },
        "guardrails": {
            "input_block_rate_pct": round(inj_blocked / max(inj_total, 1) * 100, 1),
            "out_of_scope_safe_rate_pct": round(oos_safe / max(oos_total, 1) * 100, 1),
            "hallucination_rate_semantic_pct": round(hall_sem, 1),
            "hallucination_rate_hybrid_pct": round(hall_hyb, 1),
            "injection_tests": inj_total,
            "injection_blocked": inj_blocked,
        },
        "latency": {
            "retrieval_p95_ms": round(_p95(sem_lat), 2),
            "pipeline_p95_ms": round(_p95(hyb_lat), 2),
            "semantic_avg_ms": round(avg(sem_lat), 2),
            "hybrid_avg_ms": round(avg(hyb_lat), 2),
            "avg_prompt_tokens": round(avg([float(x) for x in prompt_toks]), 1) if prompt_toks else 0,
            "avg_completion_tokens": round(avg([float(x) for x in completion_toks]), 1) if completion_toks else 0,
            "est_cost_per_query_usd": round(
                _estimate_cost_usd(
                    int(avg([float(x) for x in prompt_toks])) if prompt_toks else 500,
                    int(avg([float(x) for x in completion_toks])) if completion_toks else 200,
                ),
                6,
            ),
        },
        "overall_score": round(overall, 4),
        "models": {
            "embedding": EMBEDDING_MODEL,
            "reranker": RERANKER_MODEL,
            "llm": GROQ_MODEL,
        },
        "per_case_regulation": per_reg,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    p(f"Saved {RESULTS_JSON}")

    m = metrics_final
    sfx = os.getenv("EVAL_PNG_SUFFIX", "")
    plot_ragas_bars(metrics_final, proxies, EVAL_DIR / f"eval_ragas_metrics{sfx}.png")
    plot_ablation(summary, EVAL_DIR / f"eval_ablation_comparison{sfx}.png")
    plot_overall_scorecard(summary, EVAL_DIR / f"eval_overall_scorecard{sfx}.png")
    plot_latency(per_reg, EVAL_DIR / f"eval_latency_distribution{sfx}.png")
    plot_guardrails(summary["guardrails"], EVAL_DIR / f"eval_guardrails{sfx}.png")

    # Legacy path for README
    legacy = ROOT / "output" / "rag_evaluation_comparison.png"
    plot_ablation(summary, legacy)

    p("\n=== OVERALL PERFORMANCE ===")
    p(f"Faithfulness:       {m['faithfulness']:.4f}")
    p(f"Answer relevancy:   {m['answer_relevancy']:.4f}")
    p(f"Context precision:  {m['context_precision']:.4f}")
    p(f"Context recall:     {m['context_recall']:.4f}")
    p(f"Overall score:      {overall:.4f}")
    p(f"Hybrid vs semantic context recall: +{lift_pct:.1f}%")
    p(f"Outputs: {EVAL_DIR}")


if __name__ == "__main__":
    main()
