"""PDF no-silent-loss QA — TRD_LEVEL3.md §12-13, Instructions §11."""

from pathlib import Path

import pymupdf
import pytest

from packages.ingestion.manifest import get_source
from packages.ingestion.pipeline import REPO_ROOT
from packages.ingestion.qa import build_extraction_report


def _un_r94_path() -> Path:
    meta = get_source("unece-un-r94")
    return Path(REPO_ROOT / meta["canonical_path"])


def test_build_extraction_report_on_real_pdf_is_structurally_valid() -> None:
    report = build_extraction_report(str(_un_r94_path()), "irrelevant-hash-for-this-test", max_pages=5)

    assert report.status in ("PASS", "PASS_WITH_WARNINGS", "NEEDS_REVIEW", "FAIL")
    assert report.original_page_count >= 5
    assert report.processed_page_count == 5
    assert report.failed_pages == []
    assert report.pages_ocr == 0  # OCR is never executed
    assert report.ocr_engine == "NOT_AVAILABLE"
    assert report.engine == "pymupdf"
    assert report.engine_version


def test_build_extraction_report_unopenable_pdf_is_fail(tmp_path: Path) -> None:
    garbage = tmp_path / "not_a_real.pdf"
    garbage.write_bytes(b"this is not a PDF file at all, just garbage bytes")

    report = build_extraction_report(str(garbage), "irrelevant-hash", max_pages=5)

    assert report.status == "FAIL"
    assert report.failed_pages == [-1]
    assert report.processed_page_count == 0


def test_build_extraction_report_isolates_one_failed_page_not_the_whole_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core no-silent-loss guarantee: a single corrupted page must not
    hide the status of every other page in the same PDF."""
    original_get_text = pymupdf.Page.get_text

    def flaky_get_text(self: pymupdf.Page, *args: object, **kwargs: object) -> object:
        if self.number == 2:  # 0-indexed -> page_number 3
            raise RuntimeError("simulated page corruption")
        return original_get_text(self, *args, **kwargs)  # type: ignore[no-untyped-call]

    monkeypatch.setattr(pymupdf.Page, "get_text", flaky_get_text)

    report = build_extraction_report(str(_un_r94_path()), "irrelevant-hash-for-this-test", max_pages=5)

    assert report.failed_pages == [3]
    assert report.processed_page_count == 4
    assert report.status == "NEEDS_REVIEW"


def test_build_extraction_report_all_pages_failing_is_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fails(self: pymupdf.Page, *args: object, **kwargs: object) -> object:
        raise RuntimeError("every page is corrupted")

    monkeypatch.setattr(pymupdf.Page, "get_text", always_fails)

    report = build_extraction_report(str(_un_r94_path()), "irrelevant-hash-for-this-test", max_pages=3)

    assert report.status == "FAIL"
    assert len(report.failed_pages) == 3
