# ADR-0013: PDF no-silent-loss QA report + PyMuPDF structural extraction

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`TRD_LEVEL3.md` §12-13/§16-17 and Instructions §11 require every PDF to get
an `extraction_report.json` with an honest per-page accounting, and tables/
figures to stop being silently flattened into prose text (`docs/ADR/0006`'s
V1 scope cut). `docs/ADR/0011` already deferred Docling itself (disk
headroom). This ADR is the follow-up: what V1's PyMuPDF-only pipeline can
still honestly deliver without it.

## Decision

**Structural extraction** (`packages/ingestion/structure.py`): uses
PyMuPDF's own `page.find_tables()` and `page.get_images()` — real
structural detection already in the dependency this project has, not an
invented heuristic and not a Docling substitute. Verified against the real
corpus, not assumed: a full scan of `UN_R94.pdf` found 17 tables and 54
figures; the LS-DYNA theory/examples manuals' first 30 pages happened to
have none (their front matter), which is why the tests use `UN_R94.pdf`
specifically — asserting genuine found content, not merely "didn't crash."

**No-silent-loss QA** (`packages/ingestion/qa.py`): `build_extraction_report()`
does its own page-by-page loop with a `try`/`except` around *each page*,
deliberately not reusing `extract.py`'s `extract_pages()` (which raises on
the first bad page and aborts the whole document — fine for V1's "ingest
succeeds or doesn't", wrong for a QA report whose entire point is that one
bad page must never hide the status of every other page). Verified with a
real fault-injection test (`monkeypatch` forcing one specific page of a
real PDF to throw) that exactly that page — and only that page — lands in
`failed_pages`, with `status` correctly downgrading to `NEEDS_REVIEW` (not
`FAIL`, since the document as a whole is still usable).

`ingest_document()` now builds and writes the report *before* deciding to
ingest anything, and refuses (raises) rather than silently creating an
empty `READY` revision if `status == "FAIL"` (the PDF couldn't be opened at
all).

**OCR**: unchanged from `docs/ADR/0006`/`0011` — `ocr_engine: "NOT_AVAILABLE"`
recorded explicitly in every report, never silently omitted, never executed.

## Consequences

- `DocumentTable`/`DocumentFigure` are no longer always-empty tables —
  `ingest_document()` persists real rows with `page_id` traceable back to
  the source page, table content written to `data/artifacts/tables/`,
  figure images to `data/artifacts/figures/` (TRD_LEVEL3.md §14: store
  images outside PostgreSQL, metadata inside).
- `build_extraction_report()` duplicates some of `extract_pages()`'s
  per-page work (text density scoring) rather than sharing a single pass —
  an accepted, documented tradeoff: the two need genuinely different fault-
  handling semantics (abort-on-first-failure vs. isolate-and-continue), and
  a real corpus-scale profiling pass would be needed before this is worth
  unifying (`TRD_LEVEL3.md` §45: revisit only once measured).
- `extract.py`'s `_text_quality` was renamed `text_quality` (made public)
  so `qa.py` reuses the same heuristic instead of a second copy.
