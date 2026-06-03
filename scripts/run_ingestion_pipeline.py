#!/usr/bin/env python3
"""
Full ingestion pipeline:
  1. PDF -> Markdown (PaddleOCR: low-DPI cache + batched pages, or Docling)
  2. Hierarchical chunking from Markdown
  3. Embedding with BAAI/bge-base-en-v1.5
  4. Ready for hybrid BM25 + semantic retrieval

Usage:
  python scripts/run_ingestion_pipeline.py
  python scripts/run_ingestion_pipeline.py --skip-docling   # reuse markdown
  python scripts/run_ingestion_pipeline.py --only UN_R14 UN_R16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Docling ingestion pipeline")
    parser.add_argument("--skip-docling", action="store_true", help="Skip PDF conversion")
    parser.add_argument("--skip-chunk", action="store_true", help="Skip hierarchical chunking")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding")
    parser.add_argument("--force-docling", action="store_true", help="Reconvert all PDFs")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Process only PDFs/markdown matching these regulation keys (e.g. UN_R14)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  AutoSafety RAG — Ingestion Pipeline (PaddleOCR / Docling)")
    print("=" * 60)

    if not args.skip_docling:
        from data.docling_converter import run as run_docling

        print("\n[1/3] PDF -> Markdown (OCR)")
        manifest = run_docling(force=args.force_docling, only_regs=args.only)
        if manifest.get("errors"):
            print(f"  Warning: {len(manifest['errors'])} conversion errors")
    else:
        print("\n[1/3] Skipped Docling")

    if not args.skip_chunk:
        from data.hierarchical_chunker import run as run_chunk

        print("\n[2/3] Hierarchical chunking")
        run_chunk(only_regs=args.only)
    else:
        print("\n[2/3] Skipped chunking")

    if not args.skip_embed:
        from data.embed_chunks import run as run_embed

        print("\n[3/3] Embedding")
        run_embed()
    else:
        print("\n[3/3] Skipped embedding")

    from config import CHUNKS_FILE, EMBEDDINGS_FILE

    summary = {
        "chunks_file": str(CHUNKS_FILE),
        "embeddings_file": str(EMBEDDINGS_FILE),
        "chunks_exists": CHUNKS_FILE.exists(),
        "embeddings_exists": EMBEDDINGS_FILE.exists(),
    }
    if CHUNKS_FILE.exists():
        data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        summary["total_chunks"] = data.get("total_chunks", len(data.get("chunks", [])))
        summary["pipeline"] = data.get("pipeline")

    print("\n" + "=" * 60)
    print("  Pipeline complete")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print("\nRestart backend to load new artifacts:")
    print("  uvicorn backend.app.main:app --host 127.0.0.1 --port 8000")


if __name__ == "__main__":
    main()
