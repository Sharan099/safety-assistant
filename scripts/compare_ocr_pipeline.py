#!/usr/bin/env python3
"""
Compare Docling (latest) vs PaddleOCR 3.7 on the SAME PDF through the full RAG path:
  PDF -> Markdown -> hierarchical chunks -> Nomic embeddings -> retrieval eval

Outputs: output/ocr_compare/ocr_pipeline_comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OCR_BACKEND", "paddle")
os.environ.setdefault("PYMUPDF_SUPPLEMENT", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from config import EMBEDDING_MODEL  # noqa: E402
from ingestion.docling_converter import _build_converter, convert_pdf_docling, p  # noqa: E402
from ingestion.embed_chunks import embed_chunks_to_file  # noqa: E402
from ingestion.hierarchical_chunker import chunk_markdown_file, detect_regulation_type  # noqa: E402
from ingestion.paddle_ocr_converter import active_engine, convert_pdf_paddle  # noqa: E402
from backend.app.retrieval.hybrid import HybridRetriever  # noqa: E402
from backend.app.retrieval.reranker import CrossEncoderReranker  # noqa: E402

COMPARE_DIR = ROOT / "output" / "ocr_compare"
RESULTS_JSON = COMPARE_DIR / "ocr_pipeline_comparison.json"

# UN R14-focused eval questions (from tests/test_cases_20.json)
DEFAULT_QUESTIONS = [
    {
        "id": "R001",
        "question": "What are the UN R14 requirements for seat belt anchorage strength?",
        "ground_truth": "UN R14 specifies static strength tests on safety-belt anchorages with defined test loads in daN for different vehicle categories.",
    },
    {
        "id": "R002",
        "question": "What test load applies to belt anchorages in configuration 6.4.1 for M1 vehicles?",
        "ground_truth": "A test load of 1350 daN ± 20 daN is applied to traction devices on belt anchorages for M1 and N1 vehicles under section 6.4.",
    },
    {
        "id": "R016",
        "question": "What tests apply to safety belts under UN R16?",
        "ground_truth": "UN R16 covers safety belts and restraint systems including dynamic tests, geometry, buckle tests, and performance requirements.",
    },
]


def _md_header(pdf_path: Path, engine: str) -> str:
    return (
        f"---\n"
        f"source_pdf: {pdf_path.name}\n"
        f"regulation: {detect_regulation_type(pdf_path.name)}\n"
        f"converter: {engine}\n"
        f"---\n\n"
    )


def _page_count(md_text: str) -> int:
    return len(re.findall(r"^## Page \d+", md_text, re.MULTILINE))


def _numeric_density(md_text: str) -> float:
    """Share of tokens that look like loads, sections, or measurements."""
    tokens = re.findall(r"\b[\w./±]+\b", md_text.lower())
    if not tokens:
        return 0.0
    hits = sum(
        1
        for t in tokens
        if re.search(r"\d", t)
        and any(x in t for x in ("dan", "kn", "6.", "r14", "m1", "n1", "§"))
    )
    return round(hits / len(tokens), 4)


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
    return round(sum(scores) / len(scores), 4)


def _contexts_from_docs(docs: list[dict], k: int = 5) -> list[str]:
    out = []
    for d in docs[:k]:
        title = d.get("heading_path") or d.get("title", "")
        text = (d.get("text") or "")[:1200]
        out.append(f"{title}\n{text}".strip())
    return out if out else [""]


def _regulation_hit(docs: list[dict], expected: str) -> float:
    if not docs:
        return 0.0
    top = docs[0].get("regulation") or docs[0].get("document", "")
    return 1.0 if expected.upper() in str(top).upper() else 0.0


def run_extraction(method: str, pdf_path: Path, out_md: Path) -> dict:
    t0 = time.perf_counter()
    if method == "docling":
        try:
            import docling  # noqa: F401

            docling_ver = getattr(docling, "__version__", "unknown")
        except Exception:
            docling_ver = "unknown"
        converter = _build_converter()
        md, engine = convert_pdf_docling(pdf_path, converter=converter)
        lib_version = docling_ver
    elif method == "paddle":
        try:
            import paddleocr

            lib_version = getattr(paddleocr, "__version__", "unknown")
        except Exception:
            lib_version = "unknown"
        md = convert_pdf_paddle(pdf_path)
        engine = active_engine()
    else:
        raise ValueError(f"unknown method: {method}")

    elapsed = round(time.perf_counter() - t0, 1)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_md_header(pdf_path, engine) + md, encoding="utf-8")
    return {
        "method": method,
        "engine": engine,
        "library_version": lib_version,
        "markdown_path": str(out_md.relative_to(ROOT)),
        "chars": len(md),
        "pages_marked": _page_count(md),
        "numeric_token_density": _numeric_density(md),
        "seconds": elapsed,
    }


def run_chunk_embed(md_path: Path, method: str) -> dict:
    t0 = time.perf_counter()
    chunks = chunk_markdown_file(md_path)
    chunks_path = COMPARE_DIR / method / "chunks.json"
    emb_path = COMPARE_DIR / method / "embeddings.json"
    dataset = {
        "pipeline": f"{method}_hierarchical",
        "extraction_method": method,
        "total_chunks": len(chunks),
        "source_markdown": md_path.name,
        "chunks": chunks,
    }
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    embed_chunks_to_file(chunks, emb_path)
    elapsed = round(time.perf_counter() - t0, 1)
    return {
        "chunks": len(chunks),
        "chunks_path": str(chunks_path.relative_to(ROOT)),
        "embeddings_path": str(emb_path.relative_to(ROOT)),
        "chunk_embed_seconds": elapsed,
    }


def run_retrieval_eval(
    method: str,
    chunks_path: Path,
    emb_path: Path,
    questions: list[dict],
    *,
    use_reranker: bool,
) -> dict:
    retriever = HybridRetriever(chunks_file=chunks_path, embeddings_file=emb_path)
    reranker = CrossEncoderReranker() if use_reranker else None
    expected_reg = "UN_R14"
    rows = []
    latencies: list[float] = []

    for q in questions:
        t0 = time.perf_counter()
        r = retriever.retrieve(q["question"])
        docs = r.get("documents", [])
        if reranker:
            docs = reranker.rerank(q["question"], docs)["documents"]
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        contexts = _contexts_from_docs(docs)
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "context_recall_proxy": _proxy_context_recall(
                    q["question"], contexts, q["ground_truth"]
                ),
                "context_precision_proxy": _proxy_context_precision(
                    contexts, q["ground_truth"]
                ),
                "regulation_hit": _regulation_hit(docs, expected_reg),
                "top_doc_regulation": (docs[0].get("regulation") if docs else None),
                "retrieval_ms": round(ms, 1),
            }
        )

    def mean(key: str) -> float:
        vals = [row[key] for row in rows]
        return round(statistics.mean(vals), 4) if vals else 0.0

    return {
        "use_reranker": use_reranker,
        "questions": len(rows),
        "mean_context_recall_proxy": mean("context_recall_proxy"),
        "mean_context_precision_proxy": mean("context_precision_proxy"),
        "mean_regulation_hit": mean("regulation_hit"),
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else 0,
        "latency_ms_p95": round(
            sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 1
        )
        if latencies
        else 0,
        "per_question": rows,
    }


def _score_method(summary: dict) -> float:
    rag = summary["retrieval"]
    ext = summary["extraction"]
    chunk_score = min(summary["chunks"] / 50.0, 1.0) * 0.1
    return round(
        rag["mean_context_recall_proxy"] * 0.35
        + rag["mean_context_precision_proxy"] * 0.35
        + rag["mean_regulation_hit"] * 0.15
        + ext["numeric_token_density"] * 0.05
        + chunk_score,
        4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Docling vs PaddleOCR RAG pipeline")
    parser.add_argument(
        "--pdf",
        default=str(ROOT / "data" / "UN_R14.pdf"),
        help="PDF to process (same file for both methods)",
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Skip cross-encoder reranker (faster)",
    )
    parser.add_argument(
        "--methods",
        default="docling,paddle",
        help="Comma-separated: docling,paddle",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    p(f"OCR/RAG comparison on {pdf_path.name}")
    p(f"Embedding model: {EMBEDDING_MODEL}")
    p(f"Methods: {', '.join(methods)}")

    report: dict = {
        "pdf": str(pdf_path.relative_to(ROOT)),
        "embedding_model": EMBEDDING_MODEL,
        "methods": {},
    }

    for method in methods:
        p(f"\n=== {method.upper()} ===")
        out_md = COMPARE_DIR / method / f"{pdf_path.stem}.md"
        extraction = run_extraction(method, pdf_path, out_md)
        p(
            f"  extract: {extraction['chars']} chars, "
            f"{extraction['pages_marked']} pages, {extraction['seconds']}s "
            f"({extraction['engine']})"
        )

        ce = run_chunk_embed(out_md, method)
        p(f"  chunks: {ce['chunks']}, embed+chunk: {ce['chunk_embed_seconds']}s")

        chunks_path = COMPARE_DIR / method / "chunks.json"
        emb_path = COMPARE_DIR / method / "embeddings.json"
        retrieval = run_retrieval_eval(
            method,
            chunks_path,
            emb_path,
            DEFAULT_QUESTIONS,
            use_reranker=not args.skip_reranker,
        )
        p(
            f"  RAG proxies: recall={retrieval['mean_context_recall_proxy']}, "
            f"precision={retrieval['mean_context_precision_proxy']}, "
            f"reg_hit={retrieval['mean_regulation_hit']}"
        )

        summary = {
            "extraction": extraction,
            **ce,
            "retrieval": retrieval,
        }
        summary["composite_score"] = _score_method(summary)
        report["methods"][method] = summary

    ranked = sorted(
        report["methods"].items(),
        key=lambda kv: kv[1]["composite_score"],
        reverse=True,
    )
    winner = ranked[0][0]
    report["winner"] = winner
    report["recommendation"] = (
        f"Use {winner} for production ingestion "
        f"(composite_score={report['methods'][winner]['composite_score']})."
    )

    RESULTS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    p(f"\nWinner: {winner}")
    p(f"Saved -> {RESULTS_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
