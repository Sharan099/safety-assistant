"""PDF table/figure structural extraction — TRD_LEVEL3.md §16/§17,
PRD_LEVEL3.md §9.

Uses PyMuPDF's own table-finding (`page.find_tables()`) and image
enumeration (`page.get_images()`) — real structural detection already
built into the dependency this project already has, not a Docling
substitute and not an invented heuristic (`docs/ADR/0011` explains why
Docling itself is deferred).

A table PyMuPDF can't confidently extract still produces a row —
`quality_score` says how much to trust it, the source page is never lost
(TRD_LEVEL3.md §17: "Do not discard figures after text extraction").
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf


@dataclass
class TableExtraction:
    page_number: int  # 1-indexed
    table_index: int  # 0-indexed, within the page
    bbox: tuple[float, float, float, float]
    row_count: int
    col_count: int
    rows: list[list[str | None]]
    extraction_method: str
    quality_score: float  # 1.0: header + >=1 data row; 0.5: rows but no header; 0.0: extraction produced nothing


@dataclass
class FigureExtraction:
    page_number: int
    figure_index: int
    xref: int
    bbox: tuple[float, float, float, float] | None
    image_bytes: bytes
    image_ext: str


def _table_quality(rows: list[list[str | None]], header_names: tuple[str, ...] | None) -> float:
    if not rows:
        return 0.0
    if header_names and any(n for n in header_names):
        return 1.0
    return 0.5


def extract_tables(pdf_path: str, *, max_pages: int | None = None) -> list[TableExtraction]:
    tables: list[TableExtraction] = []
    with pymupdf.open(pdf_path) as doc:  # type: ignore[no-untyped-call]
        page_count = len(doc) if max_pages is None else min(max_pages, len(doc))
        for i in range(page_count):
            page = doc[i]
            try:
                finder = page.find_tables()
            except Exception:  # noqa: BLE001 — a page pymupdf's table finder chokes on still yields no tables, not a crash
                continue
            for t_idx, table in enumerate(finder.tables):
                try:
                    rows = table.extract()
                except Exception:  # noqa: BLE001
                    rows = []
                header_names = table.header.names if table.header else None
                tables.append(
                    TableExtraction(
                        page_number=i + 1,
                        table_index=t_idx,
                        bbox=tuple(table.bbox),
                        row_count=len(rows),
                        col_count=len(rows[0]) if rows else 0,
                        rows=rows,
                        extraction_method="pymupdf_find_tables",
                        quality_score=_table_quality(rows, header_names),
                    )
                )
    return tables


def extract_figures(pdf_path: str, *, max_pages: int | None = None) -> list[FigureExtraction]:
    figures: list[FigureExtraction] = []
    with pymupdf.open(pdf_path) as doc:  # type: ignore[no-untyped-call]
        page_count = len(doc) if max_pages is None else min(max_pages, len(doc))
        for i in range(page_count):
            page = doc[i]
            for f_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                except Exception:  # noqa: BLE001 — an unextractable image is skipped, not a crash
                    continue
                bbox = None
                rects = page.get_image_rects(xref)
                if rects:
                    bbox = (rects[0].x0, rects[0].y0, rects[0].x1, rects[0].y1)
                figures.append(
                    FigureExtraction(
                        page_number=i + 1,
                        figure_index=f_idx,
                        xref=xref,
                        bbox=bbox,
                        image_bytes=base["image"],
                        image_ext=base["ext"],
                    )
                )
    return figures
