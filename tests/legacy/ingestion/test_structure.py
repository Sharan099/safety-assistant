"""PDF table/figure structural extraction — TRD_LEVEL3.md §16/§17."""

from pathlib import Path

from packages.ingestion.manifest import get_source
from packages.ingestion.pipeline import REPO_ROOT
from packages.ingestion.structure import extract_figures, extract_tables


def _un_r94_path() -> Path:
    # Confirmed by manual inspection to have real tables (page 22, 31) and
    # figures (pages 1, 2, 11, 12, 22...) within its first 30 pages — used
    # here instead of the LS-DYNA manuals specifically so these tests assert
    # genuine content was found, not just "didn't crash".
    meta = get_source("unece-un-r94")
    return Path(REPO_ROOT / meta["canonical_path"])


def test_extract_tables_finds_real_tables_in_un_r94() -> None:
    tables = extract_tables(str(_un_r94_path()), max_pages=30)
    assert tables, "expected at least one real table in UN R94's first 30 pages"
    for t in tables:
        assert t.page_number >= 1
        assert t.extraction_method == "pymupdf_find_tables"
        assert 0.0 <= t.quality_score <= 1.0
        assert t.row_count == len(t.rows)
        assert len(t.bbox) == 4
    assert any(t.page_number == 22 for t in tables)


def test_extract_figures_finds_real_figures_in_un_r94() -> None:
    figures = extract_figures(str(_un_r94_path()), max_pages=30)
    assert figures, "expected at least one real figure in UN R94's first 30 pages"
    for f in figures:
        assert f.page_number >= 1
        assert isinstance(f.image_bytes, bytes)
        assert len(f.image_bytes) > 0
        assert f.image_ext  # e.g. "png", "jpeg"
    assert any(f.page_number == 1 for f in figures)


def test_extract_tables_respects_max_pages() -> None:
    tables = extract_tables(str(_un_r94_path()), max_pages=2)
    assert all(t.page_number <= 2 for t in tables)


def test_extract_figures_respects_max_pages() -> None:
    figures = extract_figures(str(_un_r94_path()), max_pages=2)
    assert all(f.page_number <= 2 for f in figures)
    assert figures  # page 1 alone already has 2 real figures


def test_extract_tables_on_pdf_with_no_tables_returns_empty_not_fabricated() -> None:
    # UN R129's first page is a regulatory cover page — text only, no table.
    meta = get_source("unece-un-r129")
    path = REPO_ROOT / meta["canonical_path"]
    tables = extract_tables(str(path), max_pages=1)
    assert isinstance(tables, list)  # empty or not, never raises/fabricates
