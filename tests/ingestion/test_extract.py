from packages.ingestion.extract import extract_pages
from packages.ingestion.manifest import get_source
from packages.ingestion.pipeline import REPO_ROOT


def test_extract_un_r94_first_pages() -> None:
    meta = get_source("unece-un-r94")
    pdf_path = REPO_ROOT / meta["canonical_path"]
    pages = extract_pages(str(pdf_path), max_pages=5)

    assert len(pages) == 5
    assert pages[0].page_number == 1
    # A real regulation PDF should have readable text, not a scanned image.
    assert any(p.text_quality > 0.1 for p in pages)
    assert all(0.0 <= p.text_quality <= 1.0 for p in pages)


def test_extract_respects_max_pages() -> None:
    meta = get_source("unece-un-r94")
    pdf_path = REPO_ROOT / meta["canonical_path"]
    pages = extract_pages(str(pdf_path), max_pages=2)
    assert len(pages) == 2
