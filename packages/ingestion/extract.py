"""PyMuPDF text extraction — TRD.md §14, ENVIRONMENT_SETUP.md §13 Stage A.

Deliberately NOT implemented in V1, per ENVIRONMENT_SETUP.md §13 (Stage C/D
are enrichment, not the baseline) and the hardware constraint (8 GB RAM, no
local VLM over the corpus):

  - OCR — pages that fail the text-quality gate are flagged
    (`PageQuality.needs_ocr`), never silently skipped or faked.
  - Table/figure/equation structural extraction (Docling's job, TRD.md §13) —
    pages are stored as plain text; a table in the PDF becomes prose-like
    text, not a `DocumentTable` row. Acceptable for V1 full-text retrieval
    over regulations/manuals; revisit when a RAG eval (TRD.md §14) shows
    table-heavy queries suffering for it.
  - VLM enrichment — nothing here invents content for a page it can't read.

Every extracted page keeps its raw PyMuPDF text and a text-quality score so
low-quality pages are visible, not silently accepted as equivalent to a
clean page (ENVIRONMENT_SETUP.md §13 Stage B/D: "Never silently discard
extraction failures").
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

# Below this many non-whitespace characters, a page's text layer is treated
# as unreliable — likely scanned/image-only or extraction failure.
MIN_CHARS_FOR_RELIABLE_TEXT = 40


@dataclass
class PageExtraction:
    page_number: int  # 1-indexed, matching how engineers cite PDF pages
    text: str
    char_count: int
    text_quality: float  # 0..1, see `text_quality()`
    needs_ocr: bool


def text_quality(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    # Simple, explainable heuristic: characters-per-page relative to a page
    # we'd expect to be "normal" prose density. Not a substitute for a real
    # layout-confidence model (Docling) — see module docstring.
    normal_density = 1500  # characters
    return min(1.0, len(stripped) / normal_density)


def extract_pages(pdf_path: str, *, max_pages: int | None = None) -> list[PageExtraction]:
    """Extract per-page plain text. `max_pages` bounds work for a smoke run
    on a large manual (ENVIRONMENT_SETUP.md §1: "process incrementally")."""
    pages: list[PageExtraction] = []
    with pymupdf.open(pdf_path) as doc:  # type: ignore[no-untyped-call]  # pymupdf stubs incomplete
        page_count = len(doc) if max_pages is None else min(max_pages, len(doc))
        for i in range(page_count):
            # PyMuPDF occasionally emits an embedded NUL byte from certain
            # font/encoding quirks — invisible in a rendered PDF, but
            # PostgreSQL's text type flatly rejects it. Found at real scale
            # (page 1001+ of a 2000-page manual, past where the old 20-page
            # bound ever reached) — stripped, not fabricated: a NUL byte was
            # never going to render as meaningful content either way.
            text = doc[i].get_text("text").replace("\x00", "")
            quality = text_quality(text)
            pages.append(
                PageExtraction(
                    page_number=i + 1,
                    text=text,
                    char_count=len(text.strip()),
                    text_quality=quality,
                    needs_ocr=len(text.strip()) < MIN_CHARS_FOR_RELIABLE_TEXT,
                )
            )
    return pages
