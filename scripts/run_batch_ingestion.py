#!/usr/bin/env python3
"""
Batch ingestion pipeline — root data/ PDFs only (excludes data/regulations/).

Per batch: OCR → chunk → incremental embed → checkpoint.
Resumable via output/batch_ingest_state.json.

Usage (conda activate rag):
  python scripts/run_batch_ingestion.py
  python scripts/run_batch_ingestion.py --batch-size 2 --fresh
  python scripts/run_batch_ingestion.py --skip-eval
  python scripts/run_batch_ingestion.py --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from config import CHUNKS_FILE, DATA_DIR, EMBEDDINGS_FILE, MARKDOWN_DIR  # noqa: E402
from data.docling_converter import (  # noqa: E402
    _build_converter,
    convert_single_pdf,
    discover_root_pdfs,
    p,
)
from data.embed_chunks import run_incremental  # noqa: E402
from data.hierarchical_chunker import chunk_markdown_file, detect_regulation_type  # noqa: E402

STATE_FILE = ROOT / "output" / "batch_ingest_state.json"
LOG_FILE = ROOT / "output" / "batch_ingest.log"


def _log(msg: str) -> None:
    p(msg)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"completed_pdfs": [], "batches_done": 0, "errors": []}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_chunks_dataset() -> dict:
    if CHUNKS_FILE.exists():
        return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return {
        "pipeline": "docling_hierarchical_batch",
        "total_chunks": 0,
        "source_markdown_files": 0,
        "chunks": [],
        "stats_by_regulation": {},
        "source_pdfs": [],
    }


def _save_chunks_dataset(dataset: dict) -> None:
    dataset["total_chunks"] = len(dataset.get("chunks", []))
    CHUNKS_FILE.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_chunks(dataset: dict, new_chunks: list[dict], md_name: str) -> None:
    existing_ids = {c["chunk_id"] for c in dataset.get("chunks", []) if c.get("chunk_id")}
    added = [c for c in new_chunks if c.get("chunk_id") not in existing_ids]
    dataset.setdefault("chunks", []).extend(added)
    stats = dataset.setdefault("stats_by_regulation", {})
    for c in added:
        reg = c.get("regulation", "UNKNOWN")
        stats[reg] = stats.get(reg, 0) + 1
    sources = set(dataset.get("source_pdfs", []))
    sources.add(md_name)
    dataset["source_pdfs"] = sorted(sources)
    dataset["source_markdown_files"] = len(sources)


def _init_fresh() -> None:
    """Reset artifacts for a root-only corpus rebuild."""
    for path in (CHUNKS_FILE, EMBEDDINGS_FILE, STATE_FILE):
        if path.exists():
            path.unlink()
    _log("Fresh start: cleared chunks, embeddings, and batch state")


def _batches(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _get_converter():
    from config import OCR_ENGINE

    if OCR_ENGINE == "paddle":
        return None
    if OCR_ENGINE == "pymupdf":
        return None
    try:
        return _build_converter()
    except Exception as exc:
        _log(f"Docling init failed: {exc} — per-file fallback")
        return None


def run_batch_pipeline(
    *,
    batch_size: int = 2,
    fresh: bool = False,
    force_ocr: bool = False,
    resume: bool = True,
) -> dict:
    if fresh:
        _init_fresh()

    state = _load_state() if resume else {"completed_pdfs": [], "batches_done": 0, "errors": []}
    completed = set(state.get("completed_pdfs", []))

    all_pdfs = discover_root_pdfs()
    pending = [p for p in all_pdfs if p.name not in completed]

    _log("=" * 60)
    _log(f"Batch ingestion — root PDFs only under {DATA_DIR}")
    _log(f"Total root PDFs: {len(all_pdfs)} | Pending: {len(pending)} | Batch size: {batch_size}")
    _log("=" * 60)

    if not pending:
        _log("All root PDFs already processed.")
        dataset = _load_chunks_dataset()
        return {"status": "complete", "total_chunks": dataset.get("total_chunks", 0), "state": state}

    converter = _get_converter()
    dataset = _load_chunks_dataset()
    embed_model = None
    batch_groups = _batches(pending, batch_size)

    for batch_idx, batch_pdfs in enumerate(batch_groups, start=state.get("batches_done", 0) + 1):
        _log(f"\n--- Batch {batch_idx}/{state.get('batches_done', 0) + len(batch_groups)} "
             f"({len(batch_pdfs)} PDFs) ---")
        batch_chunks: list[dict] = []

        # 1) OCR / PDF → Markdown
        for pdf_path in batch_pdfs:
            _log(f"  [OCR] {pdf_path.name}")
            try:
                result = convert_single_pdf(pdf_path, force=force_ocr, converter=converter)
                _log(f"    -> {result['status']} {result.get('markdown', '')} "
                     f"({result.get('seconds', 0)}s)")
            except Exception as exc:
                err = {"pdf": pdf_path.name, "stage": "ocr", "error": str(exc)}
                state.setdefault("errors", []).append(err)
                _log(f"    ERROR OCR: {exc}")
                continue

            # 2) Chunk this markdown
            md_path = MARKDOWN_DIR / f"{pdf_path.stem}.md"
            if not md_path.exists():
                _log(f"    SKIP chunk: markdown missing for {pdf_path.name}")
                continue
            try:
                chunks = chunk_markdown_file(md_path)
                _merge_chunks(dataset, chunks, md_path.name)
                batch_chunks.extend(chunks)
                _log(f"    [CHUNK] {md_path.name} -> {len(chunks)} chunks")
            except Exception as exc:
                err = {"pdf": pdf_path.name, "stage": "chunk", "error": str(exc)}
                state.setdefault("errors", []).append(err)
                _log(f"    ERROR CHUNK: {exc}")
                continue

            state.setdefault("completed_pdfs", []).append(pdf_path.name)
            completed.add(pdf_path.name)

        # Persist chunks after each batch (crash-safe)
        _save_chunks_dataset(dataset)
        _log(f"  [SAVE] {dataset['total_chunks']} total chunks on disk")

        # 3) Incremental embed for this batch's new chunks
        if batch_chunks:
            try:
                from sentence_transformers import SentenceTransformer
                from config import EMBEDDING_MODEL, EMBEDDING_TRUST_REMOTE_CODE

                if embed_model is None:
                    _log(f"  [EMBED] Loading {EMBEDDING_MODEL}...")
                    embed_model = SentenceTransformer(
                        EMBEDDING_MODEL,
                        device="cpu",
                        trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE,
                    )
                run_incremental(batch_chunks, model=embed_model)
            except Exception as exc:
                err = {"batch": batch_idx, "stage": "embed", "error": str(exc)}
                state.setdefault("errors", []).append(err)
                _log(f"  ERROR EMBED: {exc}")

        state["batches_done"] = batch_idx
        _save_state(state)
        gc.collect()
        _log(f"  Batch {batch_idx} complete. Running total: {dataset['total_chunks']} chunks")

    # Release embed model before eval
    embed_model = None
    gc.collect()

    summary = {
        "status": "complete",
        "pdfs_total": len(all_pdfs),
        "pdfs_processed": len(completed),
        "total_chunks": dataset.get("total_chunks", 0),
        "stats_by_regulation": dataset.get("stats_by_regulation", {}),
        "errors": state.get("errors", []),
    }
    _log("\n" + "=" * 60)
    _log("Batch pipeline complete")
    for k, v in summary.items():
        if k != "stats_by_regulation":
            _log(f"  {k}: {v}")
    _log(f"  regulations: {summary['stats_by_regulation']}")
    _log("=" * 60)
    return summary


def run_eval() -> None:
    env = os.environ.copy()
    env.setdefault("HF_HOME", r"H:\hf_cache")
    env["EVAL_SKIP_LLM"] = "false"
    env["EVAL_TEST_CASES"] = "tests/test_cases_20.json"
    env["EVAL_RESULTS_NAME"] = "rag_eval_20_results.json"
    env["EVAL_PNG_SUFFIX"] = "_20"
    _log("\n[EVAL] Running 10-question Groq evaluation...")
    subprocess.run(
        [sys.executable, str(ROOT / "tests" / "run_full_evaluation.py")],
        cwd=str(ROOT),
        env=env,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ingest root data/ PDFs only")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("INGEST_BATCH_SIZE", "2")))
    parser.add_argument("--fresh", action="store_true", help="Clear chunks/embeddings/state and rebuild")
    parser.add_argument("--force-ocr", action="store_true", help="Re-OCR even if markdown exists")
    parser.add_argument("--no-resume", action="store_true", help="Ignore batch state checkpoint")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-ingestion evaluation")
    args = parser.parse_args()

    LOG_FILE.write_text("", encoding="utf-8")  # fresh log per run
    t0 = time.perf_counter()
    summary = run_batch_pipeline(
        batch_size=args.batch_size,
        fresh=args.fresh,
        force_ocr=args.force_ocr,
        resume=not args.no_resume,
    )
    elapsed = round(time.perf_counter() - t0, 1)
    _log(f"Total pipeline time: {elapsed}s")

    if not args.skip_eval and summary.get("status") == "complete":
        run_eval()


if __name__ == "__main__":
    main()
