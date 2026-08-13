"""PDF no-silent-loss QA — TRD_LEVEL3.md §12-13, PRD_LEVEL3.md §8-9,
CLAUDE_CODE_LEVEL3_INSTRUCTIONS.md §11.

Every PDF gets an `ExtractionReport`. This module does its own page-by-page
loop (rather than delegating to `packages/ingestion/extract.py`'s
`extract_pages()`, which raises on the first bad page and aborts the whole
document) specifically so one corrupt/malformed page never hides the status
of every other page in the same PDF — the entire point of "no silent loss."
A page that fails is recorded in `failed_pages`, not silently skipped.

OCR is never executed here (`docs/ADR/0006`/`docs/ADR/0011`: no `tesseract`
binary on this machine) — `ocr_engine="NOT_AVAILABLE"` is recorded
explicitly rather than the field being silently absent.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import pymupdf

from packages.ingestion.extract import MIN_CHARS_FOR_RELIABLE_TEXT, text_quality
from packages.ingestion.structure import extract_figures, extract_tables

ExtractionStatus = Literal["PASS", "PASS_WITH_WARNINGS", "NEEDS_REVIEW", "FAIL"]

LOW_QUALITY_THRESHOLD = 0.3


@dataclass
class ExtractionReport:
    source_sha256: str
    original_page_count: int
    processed_page_count: int
    pages_with_text: int
    pages_without_text: int
    pages_ocr: int  # always 0 — OCR is never executed, see module docstring
    pages_with_tables: int
    pages_with_figures: int
    pages_with_warnings: int
    low_quality_pages: list[int]
    failed_pages: list[int]
    engine: str
    engine_version: str
    ocr_engine: str
    status: ExtractionStatus

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _fail_report(source_sha256: str, original_page_count: int, engine_version: str) -> ExtractionReport:
    return ExtractionReport(
        source_sha256=source_sha256,
        original_page_count=original_page_count,
        processed_page_count=0,
        pages_with_text=0,
        pages_without_text=0,
        pages_ocr=0,
        pages_with_tables=0,
        pages_with_figures=0,
        pages_with_warnings=0,
        low_quality_pages=[],
        failed_pages=list(range(1, original_page_count + 1)) if original_page_count else [-1],
        engine="pymupdf",
        engine_version=engine_version,
        ocr_engine="NOT_AVAILABLE",
        status="FAIL",
    )


def build_extraction_report(pdf_path: str, source_sha256: str, *, max_pages: int | None = None) -> ExtractionReport:
    engine_version: str = pymupdf.pymupdf_version  # type: ignore[attr-defined]  # pymupdf stubs incomplete

    try:
        doc = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]  # pymupdf stubs incomplete
    except Exception:  # noqa: BLE001 — an unopenable PDF is FAIL, not a crash
        return _fail_report(source_sha256, 0, engine_version)

    try:
        original_page_count = len(doc)
    except Exception:  # noqa: BLE001
        doc.close()  # type: ignore[no-untyped-call]
        return _fail_report(source_sha256, 0, engine_version)

    page_count = original_page_count if max_pages is None else min(max_pages, original_page_count)

    pages_with_text = 0
    pages_without_text = 0
    low_quality_pages: list[int] = []
    failed_pages: list[int] = []
    warning_pages: set[int] = set()
    table_pages: set[int] = set()
    figure_pages: set[int] = set()

    for i in range(page_count):
        page_number = i + 1
        try:
            page = doc[i]
            text = page.get_text("text")  # type: ignore[no-untyped-call]  # pymupdf stubs incomplete
            quality = text_quality(text)
            needs_ocr = len(text.strip()) < MIN_CHARS_FOR_RELIABLE_TEXT
            if needs_ocr:
                pages_without_text += 1
                warning_pages.add(page_number)
            else:
                pages_with_text += 1
            if 0 < quality < LOW_QUALITY_THRESHOLD:
                low_quality_pages.append(page_number)
                warning_pages.add(page_number)
        except Exception:  # noqa: BLE001 — this page failed; every other page still gets processed
            failed_pages.append(page_number)

    doc.close()  # type: ignore[no-untyped-call]

    try:
        table_pages = {t.page_number for t in extract_tables(pdf_path, max_pages=max_pages)}
    except Exception:  # noqa: BLE001 — table detection failing doesn't invalidate text extraction
        pass
    try:
        figure_pages = {f.page_number for f in extract_figures(pdf_path, max_pages=max_pages)}
    except Exception:  # noqa: BLE001
        pass

    processed_page_count = page_count - len(failed_pages)

    if failed_pages:
        status: ExtractionStatus = "FAIL" if len(failed_pages) == page_count else "NEEDS_REVIEW"
    elif warning_pages:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    return ExtractionReport(
        source_sha256=source_sha256,
        original_page_count=original_page_count,
        processed_page_count=processed_page_count,
        pages_with_text=pages_with_text,
        pages_without_text=pages_without_text,
        pages_ocr=0,
        pages_with_tables=len(table_pages),
        pages_with_figures=len(figure_pages),
        pages_with_warnings=len(warning_pages),
        low_quality_pages=low_quality_pages,
        failed_pages=failed_pages,
        engine="pymupdf",
        engine_version=engine_version,
        ocr_engine="NOT_AVAILABLE",
        status=status,
    )
