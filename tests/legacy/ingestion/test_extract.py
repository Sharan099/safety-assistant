import pymupdf
import pytest

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


def test_extract_strips_embedded_nul_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real bug, found only at real scale: PyMuPDF can emit an embedded
    NUL byte from certain font/encoding quirks (surfaced past page 1000 of
    a 2000-page LS-DYNA manual, invisible under the old 20-page bound) —
    PostgreSQL's text type rejects it outright. Reproduced here via fault
    injection rather than waiting for another 1000+ page real PDF."""
    original_get_text = pymupdf.Page.get_text

    def get_text_with_nul(self: pymupdf.Page, *args: object, **kwargs: object) -> object:
        real_text = original_get_text(self, *args, **kwargs)  # type: ignore[no-untyped-call]
        return real_text[:10] + "\x00" + real_text[10:]

    monkeypatch.setattr(pymupdf.Page, "get_text", get_text_with_nul)

    meta = get_source("unece-un-r94")
    pdf_path = REPO_ROOT / meta["canonical_path"]
    pages = extract_pages(str(pdf_path), max_pages=1)

    assert "\x00" not in pages[0].text
