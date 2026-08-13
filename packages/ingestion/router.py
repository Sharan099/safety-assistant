"""PDF document/page router — PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md §4/§9/§10.

"Use a router instead of one parser for every document." Only PyMuPDF is
actually installed — Docling and OCR are deferred (`docs/ADR/0011`: 12 GB
free disk, no `tesseract` binary). A page routed to `COMPLEX_LAYOUT` or
`SCANNED_IMAGE_ONLY` still gets PyMuPDF's best-effort output (there's no
other engine to hand it to), but the routing *decision* — which engine
would ideally handle this page, and the honest reason that engine wasn't
used — is recorded rather than silently collapsed into "just used PyMuPDF
for everything, as if that were the only option ever considered."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RouteDecision = Literal["SIMPLE_DIGITAL", "COMPLEX_LAYOUT", "SCANNED_IMAGE_ONLY"]
Engine = Literal["pymupdf", "docling", "ocr"]


@dataclass
class PageRoute:
    page_number: int
    decision: RouteDecision
    ideal_engine: Engine
    actual_engine: Engine
    engine_available: bool
    reason: str


def route_page(*, page_number: int, needs_ocr: bool, has_low_quality_table: bool, has_figures: bool) -> PageRoute:
    if needs_ocr:
        return PageRoute(
            page_number=page_number,
            decision="SCANNED_IMAGE_ONLY",
            ideal_engine="ocr",
            actual_engine="pymupdf",
            engine_available=False,
            reason="no tesseract binary installed (docs/ADR/0011) — NEEDS_REVIEW, never silently OCR'd or dropped",
        )
    if has_low_quality_table or has_figures:
        return PageRoute(
            page_number=page_number,
            decision="COMPLEX_LAYOUT",
            ideal_engine="docling",
            actual_engine="pymupdf",
            engine_available=False,
            reason="Docling not installed (docs/ADR/0011) — used PyMuPDF's heuristic extraction instead",
        )
    return PageRoute(
        page_number=page_number,
        decision="SIMPLE_DIGITAL",
        ideal_engine="pymupdf",
        actual_engine="pymupdf",
        engine_available=True,
        reason="native text layer, no complex structure detected",
    )


def route_document(
    *,
    total_pages: int,
    pages_needing_ocr: set[int],
    pages_with_tables: set[int],
    pages_with_figures: set[int],
) -> list[PageRoute]:
    return [
        route_page(
            page_number=i,
            needs_ocr=i in pages_needing_ocr,
            has_low_quality_table=i in pages_with_tables,
            has_figures=i in pages_with_figures,
        )
        for i in range(1, total_pages + 1)
    ]


def summarize_routes(routes: list[PageRoute]) -> dict[str, int]:
    summary: dict[str, int] = {"SIMPLE_DIGITAL": 0, "COMPLEX_LAYOUT": 0, "SCANNED_IMAGE_ONLY": 0}
    for r in routes:
        summary[r.decision] += 1
    return summary
