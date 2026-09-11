"""PDF document/page router — PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md §4/§9/§10."""

from packages.ingestion.router import route_document, route_page, summarize_routes


def test_scanned_page_routes_to_ocr_marked_unavailable() -> None:
    route = route_page(page_number=3, needs_ocr=True, has_low_quality_table=False, has_figures=False)
    assert route.decision == "SCANNED_IMAGE_ONLY"
    assert route.ideal_engine == "ocr"
    assert route.actual_engine == "pymupdf"
    assert route.engine_available is False
    assert "tesseract" in route.reason


def test_table_heavy_page_routes_to_docling_marked_unavailable() -> None:
    route = route_page(page_number=5, needs_ocr=False, has_low_quality_table=True, has_figures=False)
    assert route.decision == "COMPLEX_LAYOUT"
    assert route.ideal_engine == "docling"
    assert route.engine_available is False
    assert "Docling" in route.reason


def test_figure_page_also_routes_to_complex_layout() -> None:
    route = route_page(page_number=7, needs_ocr=False, has_low_quality_table=False, has_figures=True)
    assert route.decision == "COMPLEX_LAYOUT"


def test_plain_text_page_routes_to_simple_digital_and_engine_is_available() -> None:
    route = route_page(page_number=1, needs_ocr=False, has_low_quality_table=False, has_figures=False)
    assert route.decision == "SIMPLE_DIGITAL"
    assert route.ideal_engine == "pymupdf"
    assert route.engine_available is True


def test_ocr_takes_priority_over_complex_layout_when_both_true() -> None:
    # A scanned page with an image that happens to look like a table
    # doesn't have real table structure to route to Docling for — OCR is
    # the actual gap.
    route = route_page(page_number=2, needs_ocr=True, has_low_quality_table=True, has_figures=True)
    assert route.decision == "SCANNED_IMAGE_ONLY"


def test_route_document_produces_one_route_per_page() -> None:
    routes = route_document(total_pages=5, pages_needing_ocr={2}, pages_with_tables={4}, pages_with_figures=set())
    assert [r.page_number for r in routes] == [1, 2, 3, 4, 5]
    assert routes[1].decision == "SCANNED_IMAGE_ONLY"
    assert routes[3].decision == "COMPLEX_LAYOUT"
    assert routes[0].decision == "SIMPLE_DIGITAL"


def test_summarize_routes_counts_each_decision() -> None:
    routes = route_document(total_pages=4, pages_needing_ocr={1}, pages_with_tables={2}, pages_with_figures=set())
    summary = summarize_routes(routes)
    assert summary == {"SIMPLE_DIGITAL": 2, "COMPLEX_LAYOUT": 1, "SCANNED_IMAGE_ONLY": 1}
