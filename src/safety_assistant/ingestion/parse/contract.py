"""Parser contract. The rest of ingestion depends on these types only —
PyMuPDF today, Docling/OCR tomorrow, behind the same Protocol."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed, as engineers cite pages
    text: str
    char_count: int
    text_quality: float  # 0..1 heuristic density
    needs_ocr: bool
    # Which engine would ideally handle this page vs which did — recorded, never hidden.
    route: str = "SIMPLE_DIGITAL"  # SIMPLE_DIGITAL | COMPLEX_LAYOUT | SCANNED_IMAGE_ONLY


@dataclass
class ParsedTable:
    page_number: int
    table_index: int
    bbox: tuple[float, float, float, float]
    headers: list[str | None] | None
    rows: list[list[str | None]]
    extraction_method: str
    quality_score: float  # 1.0 header + data rows; 0.5 rows, no header; 0.0 nothing


@dataclass
class ParsedFigure:
    """Image bytes are streamed to a sink during parsing (a 4,000-page manual holds
    thousands of images); the parser keeps only the content hash + storage URI."""

    page_number: int
    figure_index: int
    bbox: tuple[float, float, float, float] | None
    image_ext: str
    image_sha256: str
    storage_uri: str | None = None  # set when a FigureSink was provided
    size_bytes: int = 0


FigureSink = Callable[[bytes, str], str]  # (image bytes, extension) -> storage uri


@dataclass
class ExtractionReport:
    """No-silent-loss QA: every page is accounted for (PASS / warnings / failed)."""

    source_sha256: str
    engine: str
    engine_version: str
    ocr_engine: str
    original_page_count: int
    processed_page_count: int
    pages_with_text: int
    pages_without_text: int
    pages_with_tables: int
    pages_with_figures: int
    low_quality_pages: list[int]
    failed_pages: list[int]
    route_summary: dict[str, int]
    status: str  # PASS | PASS_WITH_WARNINGS | NEEDS_REVIEW | FAIL

    def to_dict(self) -> dict[str, object]:
        import dataclasses

        return dataclasses.asdict(self)


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    tables: list[ParsedTable] = field(default_factory=list)
    figures: list[ParsedFigure] = field(default_factory=list)
    report: ExtractionReport | None = None
    pdf_metadata: dict[str, str | None] = field(default_factory=dict)


class DocumentParser(Protocol):
    name: str
    version: str

    def config_hash(self) -> str: ...

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        source_sha256: str,
        max_pages: int | None = None,
        figure_sink: FigureSink | None = None,
    ) -> ParsedDocument: ...
