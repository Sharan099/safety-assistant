# ADR-0006: V1 ingestion pipeline scope — PyMuPDF only, no OCR/VLM/tables/figures/equations

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`TRD.md` §13-18 and `ENVIRONMENT_SETUP.md` §13 describe a fuller pipeline:
Docling as the primary structured-document parser, OCR for scanned pages, a
selective VLM enrichment pass, and dedicated table/figure/equation
extraction into their own tables. The dev machine is 8 GB RAM / 2 GB GPU
(`ENVIRONMENT_SETUP.md` §1), and `TRD.md` §14 itself says: "Start with
PyMuPDF because it is lightweight."

## Decision

V1's `packages/ingestion` implements only:

```
register -> SHA-256 -> PyMuPDF page-text extraction -> quality heuristic
-> heading-based section detection -> paragraph-packed chunking -> persist
```

Explicitly deferred, not faked:

- **Docling** — not installed. `PageQuality`/extraction-quality fields exist
  in the schema and are populated (`text_quality`, `ocr_used=False` always)
  so a future Docling-based revision is a additive, comparable
  `DocumentRevision`, not a schema change.
- **OCR** — pages below the text-density threshold are flagged
  (`needs_ocr` -> `layout_quality=0.0`), never OCR'd and never silently
  dropped.
- **VLM** — no VLM call anywhere in `packages/ingestion`.
- **Tables/figures/equations** — a table in a PDF is extracted as its raw
  text run (usually reads as jumbled prose), not a `DocumentTable` row.
  `document_tables`/`document_figures`/`document_equations` stay empty in
  V1. Acceptable for full-text retrieval over regulations/manuals; revisit
  if the RAG evaluation (`TRD.md` §14, Phase 12) shows table-heavy queries
  suffering for it.
- **Canonical Markdown** — `DocumentPage.markdown_content` stays `NULL`;
  only plain text is populated. Markdown export adds no retrieval value
  until a renderer actually consumes it.

## Section/chunk detection

Heading detection is a regex heuristic (numbered headings like "3.2.1
Contact Definitions", short ALL-CAPS lines) — not a layout model. A
mis-detected heading produces a slightly wrong section boundary, never
fabricated content, so the failure mode is benign and visible (the section
title itself is wrong, not hidden).

Chunking packs paragraphs to ~400 words per chunk and never crosses a
section boundary (TRD.md §18: "do not blindly split every PDF into
fixed-size chunks"). `token_count` is a word-count approximation, not a
real tokenizer count — revisit once an embedding model (and its tokenizer)
is chosen per TRD.md §20.

## Consequences

- Ingesting the full LS-DYNA corpus (largest PDF: 34 MB, hundreds of pages)
  synchronously is not attempted in this phase; `ingest_document()` accepts
  `max_pages` for a bounded smoke run, and full-corpus ingestion is a
  follow-up batch job (`IMPLEMENTATION_PLAN.md` Phase 4: "Do not process all
  documents simultaneously on the 8 GB machine").
- `DocumentRevision` is keyed by `(document, revision_label)` where the
  label encodes extractor version and page bound, so re-running is
  idempotent and a later real Docling pass becomes a new, comparable
  revision rather than overwriting this one.
