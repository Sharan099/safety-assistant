# ADR-0018: PDF page router — explicit routing decisions, honestly unavailable engines

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md` §4/§9/§10 calls for "a router instead
of one parser for every document": simple digital pages to PyMuPDF, complex
layout to Docling, scanned/image-only to OCR.

Only PyMuPDF is actually installed on this machine (`docs/ADR/0011`:
Docling needs several GB of `torch`/`transformers` against 12 GB free
disk; no `tesseract` binary exists for OCR). Building a literal
multi-engine router when two of the three engines don't exist would be
either dead code or a misleading pretense.

## Decision

`packages/ingestion/router.py` implements the real, useful half of a
router: a pure classification function (`route_page`) that decides, per
page, which engine *should* handle it (`SIMPLE_DIGITAL` → PyMuPDF,
`COMPLEX_LAYOUT` → Docling, `SCANNED_IMAGE_ONLY` → OCR) and records that
decision alongside which engine *actually* handled it and why, when they
differ. Every page still gets PyMuPDF's best-effort extraction (there's
nothing else to hand it to) — the router's contribution is making the gap
between "ideal" and "actual" a first-class, queryable fact
(`PageRoute.engine_available`, `.reason`) instead of a silently-collapsed
assumption that PyMuPDF was always the intended engine.

Wired into `packages/ingestion/qa.py`'s `ExtractionReport` as
`route_summary` (counts per decision) — visible in every
`extraction_report.json`, not a standalone unused module.

OCR routing takes priority over complex-layout routing when a page
qualifies for both (a scanned page with something that visually resembles
a table has no real extractable table structure to hand to Docling for —
OCR is the actual gap).

## Consequences

- `extraction_report.json` now answers "how many pages on this document
  would have benefited from Docling/OCR if it were installed?" — real
  signal for prioritizing when to actually install them, backed by
  measurement rather than a general sense that "some PDFs are probably
  complex."
- No behavior changes to what gets extracted today — this is purely an
  observability/honesty layer on top of the existing PyMuPDF-only pipeline.
- When Docling/OCR are eventually installed, `route_page`'s decision logic
  becomes the actual dispatch logic (call the ideal engine when
  `engine_available` would be `True`) — the classification work doesn't
  need to be redone, only the "always PyMuPDF" fallback branch changes.
