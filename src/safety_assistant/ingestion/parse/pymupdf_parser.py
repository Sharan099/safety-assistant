"""PyMuPDF implementation of `DocumentParser`.

Behaviour carried over from the baseline (proven on 10,309 real pages):

- per-page fault isolation: one bad page is recorded in `failed_pages`, the
  rest still parse — never abort the document, never fabricate;
- embedded NUL bytes stripped (PostgreSQL text rejects them; found at page 1001+
  of a real manual);
- text-quality heuristic and `needs_ocr` flag; OCR is *not* executed and the
  report says so (`ocr_engine="NOT_AVAILABLE"`);
- native table detection (`find_tables`) with header capture; figures via
  `get_images`; both best-effort per page.
- page routing decision recorded (SIMPLE_DIGITAL / COMPLEX_LAYOUT / SCANNED_IMAGE_ONLY).
"""

from __future__ import annotations

import hashlib
import json

import pymupdf

from safety_assistant.ingestion.parse.contract import (
    ExtractionReport,
    FigureSink,
    ParsedDocument,
    ParsedFigure,
    ParsedPage,
    ParsedTable,
)
from safety_assistant.ingestion.parse.ocr import OcrAdapter, OcrUnavailable

MIN_CHARS_FOR_RELIABLE_TEXT = 40
LOW_QUALITY_THRESHOLD = 0.3
_NORMAL_DENSITY_CHARS = 1500


def text_quality(text: str) -> float:
    stripped = text.strip()
    return min(1.0, len(stripped) / _NORMAL_DENSITY_CHARS) if stripped else 0.0


def _route(*, needs_ocr: bool, has_table: bool, has_figure: bool) -> str:
    if needs_ocr:
        return "SCANNED_IMAGE_ONLY"
    if has_table or has_figure:
        return "COMPLEX_LAYOUT"
    return "SIMPLE_DIGITAL"


class PyMuPDFParser:
    name = "pymupdf"
    version = "2.0.0"  # bump when extraction semantics change → re-parse is triggered by config hash

    def __init__(
        self, *, extract_tables: bool = True, extract_figures: bool = True, ocr: OcrAdapter | None = None
    ) -> None:
        self.extract_tables = extract_tables
        self.extract_figures = extract_figures
        self.ocr = ocr  # None = pages without a text layer stay flagged (NEEDS_REVIEW)

    def config_hash(self) -> str:
        cfg = {
            "parser": self.name,
            "version": self.version,
            "pymupdf": pymupdf.pymupdf_version,  # type: ignore[attr-defined]
            "tables": self.extract_tables,
            "figures": self.extract_figures,
            "ocr": self.ocr.name if self.ocr else "none",
            "min_chars": MIN_CHARS_FOR_RELIABLE_TEXT,
        }
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        source_sha256: str,
        max_pages: int | None = None,
        figure_sink: FigureSink | None = None,
    ) -> ParsedDocument:
        engine_version: str = pymupdf.pymupdf_version  # type: ignore[attr-defined]
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[no-untyped-call]
            original_count = len(doc)
        except Exception:  # noqa: BLE001 — unopenable is FAIL, not a crash
            report = ExtractionReport(
                source_sha256=source_sha256,
                engine=self.name,
                engine_version=engine_version,
                ocr_engine="NOT_AVAILABLE",
                original_page_count=0,
                processed_page_count=0,
                pages_with_text=0,
                pages_without_text=0,
                pages_with_tables=0,
                pages_with_figures=0,
                low_quality_pages=[],
                failed_pages=[-1],
                route_summary={},
                status="FAIL",
            )
            return ParsedDocument(pages=[], report=report)

        page_count = original_count if max_pages is None else min(max_pages, original_count)
        pages: list[ParsedPage] = []
        tables: list[ParsedTable] = []
        figures: list[ParsedFigure] = []
        failed: list[int] = []
        low_quality: list[int] = []

        with doc:
            for i in range(page_count):
                pn = i + 1
                try:
                    page = doc[i]
                    raw: str = page.get_text("text")  # type: ignore[no-untyped-call]
                    text = raw.replace("\x00", "")
                except Exception:  # noqa: BLE001 — isolate this page's failure
                    failed.append(pn)
                    continue
                quality = text_quality(text)
                needs_ocr = len(text.strip()) < MIN_CHARS_FOR_RELIABLE_TEXT
                if needs_ocr and self.ocr is not None:
                    try:
                        png = page.get_pixmap(dpi=200).tobytes("png")  # type: ignore[no-untyped-call]
                        text = self.ocr.ocr_png(png).replace("\x00", "")
                        quality = text_quality(text)
                        needs_ocr = len(text.strip()) < MIN_CHARS_FOR_RELIABLE_TEXT
                    except OcrUnavailable:
                        pass  # stays flagged; the report says NEEDS_REVIEW
                if 0 < quality < LOW_QUALITY_THRESHOLD:
                    low_quality.append(pn)

                page_tables = self._tables(page, pn) if self.extract_tables else []
                page_figures = self._figures(doc, page, pn, figure_sink) if self.extract_figures else []
                tables.extend(page_tables)
                figures.extend(page_figures)
                pages.append(
                    ParsedPage(
                        page_number=pn,
                        text=text,
                        char_count=len(text.strip()),
                        text_quality=quality,
                        needs_ocr=needs_ocr,
                        route=_route(needs_ocr=needs_ocr, has_table=bool(page_tables), has_figure=bool(page_figures)),
                    )
                )
            meta = {k: (doc.metadata or {}).get(k) for k in ("title", "author", "creationDate", "modDate")}

        without_text = sum(1 for p in pages if p.needs_ocr)
        route_summary: dict[str, int] = {"SIMPLE_DIGITAL": 0, "COMPLEX_LAYOUT": 0, "SCANNED_IMAGE_ONLY": 0}
        for p in pages:
            route_summary[p.route] += 1
        if failed:
            status = "FAIL" if len(failed) == page_count else "NEEDS_REVIEW"
        elif pages and without_text == len(pages):
            status = "NEEDS_REVIEW"  # scanned document and no OCR: nothing indexable came out
        elif without_text or low_quality:
            status = "PASS_WITH_WARNINGS"
        else:
            status = "PASS"
        report = ExtractionReport(
            source_sha256=source_sha256,
            engine=self.name,
            engine_version=engine_version,
            ocr_engine="NOT_AVAILABLE",
            original_page_count=original_count,
            processed_page_count=page_count - len(failed),
            pages_with_text=len(pages) - without_text,
            pages_without_text=without_text,
            pages_with_tables=len({t.page_number for t in tables}),
            pages_with_figures=len({f.page_number for f in figures}),
            low_quality_pages=low_quality,
            failed_pages=failed,
            route_summary=route_summary,
            status=status,
        )
        return ParsedDocument(pages=pages, tables=tables, figures=figures, report=report, pdf_metadata=meta)

    @staticmethod
    def _tables(page: pymupdf.Page, pn: int) -> list[ParsedTable]:
        out: list[ParsedTable] = []
        try:
            finder = page.find_tables()  # type: ignore[no-untyped-call]
        except Exception:  # noqa: BLE001
            return out
        for idx, table in enumerate(finder.tables):
            try:
                rows = table.extract()
            except Exception:  # noqa: BLE001
                rows = []
            rows = [[(c.replace("\x00", "") if isinstance(c, str) else c) for c in r] for r in rows]
            headers = list(table.header.names) if table.header else None
            has_header = bool(headers and any(h for h in headers))
            out.append(
                ParsedTable(
                    page_number=pn,
                    table_index=idx,
                    bbox=tuple(table.bbox),
                    headers=headers if has_header else None,
                    rows=rows,
                    extraction_method="pymupdf_find_tables",
                    quality_score=0.0 if not rows else (1.0 if has_header else 0.5),
                )
            )
        return out

    @staticmethod
    def _figures(doc: pymupdf.Document, page: pymupdf.Page, pn: int, sink: FigureSink | None) -> list[ParsedFigure]:
        out: list[ParsedFigure] = []
        for idx, img in enumerate(page.get_images(full=True)):  # type: ignore[no-untyped-call]
            xref = img[0]
            try:
                base = doc.extract_image(xref)  # type: ignore[no-untyped-call]
            except Exception:  # noqa: BLE001
                continue
            data: bytes = base["image"]
            ext: str = base["ext"]
            rects = page.get_image_rects(xref)
            bbox = (rects[0].x0, rects[0].y0, rects[0].x1, rects[0].y1) if rects else None
            out.append(
                ParsedFigure(
                    page_number=pn,
                    figure_index=idx,
                    bbox=bbox,
                    image_ext=ext,
                    image_sha256=hashlib.sha256(data).hexdigest(),
                    storage_uri=sink(data, ext) if sink else None,
                    size_bytes=len(data),
                )
            )
            del data  # never accumulate image bytes across pages
        return out
