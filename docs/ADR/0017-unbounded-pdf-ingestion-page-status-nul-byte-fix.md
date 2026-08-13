# ADR-0017: Removed the artificial PDF page bound, added per-page status, fixed a real NUL-byte bug found only at scale

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md` §7 calls the 20-page ingestion bound
out as an artificial limit to remove, with explicit per-page state tracking
(`DISCOVERED`/`EXTRACTED`/`OCR_REQUIRED`/.../`VALIDATED`/`FAILED`) and a
no-silent-loss rule: original page count must equal accounted page count.

Investigation showed the 20-page cap was never an architectural limit —
`packages/ingestion/pipeline.py`'s `ingest_document()` already defaults
`max_pages=None` (unbounded); the cap only ever lived in
`scripts/ingest_level3_pdfs.py`'s own `MAX_PAGES = 20` constant, set
during the Level-3 smoke-test phase to bound runtime, not memory.

## Decision

1. **Page status** (`packages/domain/knowledge.py`): `DocumentPage.status`
   (migration `6e0faa0ca413`, additive, `server_default='EXTRACTED'` for
   existing rows — every one of them genuinely did complete extraction, so
   that backfill is accurate, not a placeholder). Collapsed to
   `DISCOVERED`/`EXTRACTED`/`NEEDS_REVIEW`/`FAILED` — the doc's fuller list
   (`OCR_REQUIRED`, `OCR_COMPLETE`, `VISUAL_REVIEW_REQUIRED`) would be
   indistinguishable from `NEEDS_REVIEW` on this pipeline (no OCR/VLM
   engine installed, `docs/ADR/0011`) — not invented.

2. **Cap removed**: `scripts/ingest_level3_pdfs.py` calls
   `ingest_document(session, source_id)` with no `max_pages`, and verifies
   `original_page_count == accounted page count` for every PDF (§18's
   no-silent-loss rule, checked in code, not just asserted in prose).

3. **Real bug found only by actually removing the bound**: ingesting a full
   ~2000-page LS-DYNA manual past page 1000 hit
   `psycopg.DataError: PostgreSQL text fields cannot contain NUL (0x00)
   bytes` — PyMuPDF occasionally emits an embedded NUL byte from certain
   font/encoding quirks. Invisible under the old 20-page bound (never
   reached page 1000+ of any document). Fixed in
   `packages/ingestion/extract.py`: `.replace("\x00", "")` on every page's
   extracted text — a NUL byte was never going to render as meaningful
   content either way, so stripping it is not information loss.
   Regression-tested via fault injection (`monkeypatch` forcing
   `pymupdf.Page.get_text` to return NUL-containing text), not by waiting
   to hit another 1000+ page document.

4. **A second, unrelated crash surfaced by the same run**: the script's own
   `except Exception as exc: print(...)` handler crashed with
   `UnicodeEncodeError` — the Windows console's default codepage (cp1252)
   can't encode every Unicode character (e.g. `ﬁ`, the "fi" ligature)
   real PDF text can contain. Fixed with a `_safe_print()` helper
   (encode-with-replace) — a crash while *reporting* a failure violates
   "one bad PDF must not abort the batch" exactly as much as a crash during
   ingestion itself.

## Consequences

- Full-length ingestion of large real manuals (up to ~2000 pages) is now
  the standing behavior, not a 20-page preview — real chunk/embedding
  counts grow substantially for the LS-DYNA manuals specifically.
- `DocumentPage.status` is available for future querying/filtering (e.g.
  "show me every NEEDS_REVIEW page across the corpus") without needing to
  re-derive it from `text_quality`/`ocr_used` each time.
- Both bugs were found by actually running the real, unbounded pipeline
  against real large documents — neither would have been caught by any
  fixture or by the previous 20-page-bounded runs, reinforcing this
  project's standing practice of running real corpus content rather than
  trusting synthetic coverage alone.
