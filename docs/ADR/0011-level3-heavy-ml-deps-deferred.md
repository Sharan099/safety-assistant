# ADR-0011: Docling, OCR engine, and a real cross-encoder reranker deferred — disk headroom, not capability doubt

- **Status:** Accepted (interim — expected to be superseded, same posture as ADR-0007)
- **Date:** 2026-08-13

## Context

`TRD_LEVEL3.md` §10/§15/§25 and `CLAUDE_CODE_LEVEL3_INSTRUCTIONS.md` §10/§13/§21
name Docling as the primary PDF structural parser, tesseract-class OCR as a
fallback, and a reranker after RRF.

Measured on the actual dev machine before deciding:

- **Disk**: `H:` has **12 GB free of 239 GB (96% full)**. `uv pip install --dry-run docling`
  resolves ~30 new packages including `torch==2.13.0`, `torchvision`,
  `transformers` — multiple GB. `sentence-transformers` (needed for a real
  cross-encoder reranker) resolves the same `torch`/`transformers` stack
  again (24 new packages). The `Knowledge source/` corpus itself is 1.3 GB
  and derived artifacts (extracted text, Parquet features, archive members)
  will grow disk usage further during Level-3 processing.
- **OCR**: no `tesseract` binary is present on this machine (`which tesseract`
  → not found).
- Both `TRD_LEVEL3.md` §45 ("The application must work on the development
  machine without requiring GPU inference... CPU-friendly models") and
  `TRD_LEVEL3.md` §25 ("Do not use a model that makes the application
  unusable on the 8 GB machine") make hardware fit for the *actual* dev
  machine — not code neatness — the deciding factor.

This is the identical situation `ADR-0007` already resolved for embeddings:
a real model is correct in principle, but not installable today without
risk, and the interface should be ready for it regardless.

## Decision

Same pattern as `ADR-0007`, applied to three components:

1. **PDF structural extraction**: PyMuPDF stays primary (`ADR-0006`). Add
   heuristic table/figure detection using PyMuPDF's own `get_images()` /
   drawing-rect APIs (no new dependency) so `DocumentTable`/`DocumentFigure`
   are no longer always-empty, and a real `extraction_report.json` per PDF
   (Instructions §11). Docling becomes a documented, `Protocol`-based swap
   point (`PDFExtractionProvider`), status `NOT_INSTALLED`, not attempted
   silently and not claimed as active.
2. **OCR**: unchanged from `ADR-0006` — pages below the text-density
   threshold get `needs_ocr=True` and `layout_quality=0.0`, never OCR'd.
   The extraction report explicitly records `ocr_engine: NOT_AVAILABLE`
   rather than omitting the field.
3. **Reranker**: ships a lightweight, dependency-free `Reranker` `Protocol`
   with a default lexical/authority-aware implementation (term-overlap +
   authority-tier boost over the already-fused RRF candidate set — no new
   package). A real cross-encoder (`sentence-transformers`) is a documented,
   swappable future implementation of the same `Protocol`, deferred until
   disk headroom exists and `TRD_LEVEL3.md` §25's CPU-latency benchmark is
   actually run.

`rank-bm25` (the real BM25 implementation, see the RAG ADR) is **not**
deferred — it is a ~50 KB pure-Python package depending only on the
already-installed `numpy`, no disk-risk trade-off applies.

## Consequences

- Table/figure/equation extraction quality will be heuristic, not
  Docling-grade, until this is revisited. Every table/figure this heuristic
  can't confidently extract is marked `NEEDS_REVIEW` per the no-silent-loss
  rule, never silently dropped or fabricated.
- The reranker's lexical/authority scoring is weaker than a trained
  cross-encoder — expected, documented, and swappable without changing
  anything above `packages/retrieval/rerank.py`.
- Before either deferred component is installed, re-run `df -h` (or the
  Windows equivalent) and re-check `tesseract` availability; this ADR's
  facts, not just its conclusion, should be re-verified rather than assumed
  stale-safe.
