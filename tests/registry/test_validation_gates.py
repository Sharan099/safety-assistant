"""Unit tests for acquisition validation and tracker gates (Phase B)."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from crawler.allowlist import DomainNotAllowedError, assert_url_allowed
from crawler.change_check import remote_changed
from registry.change_tracker import ChangeTracker
from registry.validation import ValidationOutcome, validate_pdf


def _write_min_pdf(path: Path, text: str, *, pages: int = 5) -> None:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text + "\n" + ("Padding content for size. " * 80), fontsize=11)
    doc.save(path)
    doc.close()


def test_html_saved_as_pdf_rejected(tmp_path):
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"<html><body>UN Regulation No. 94</body></html>")
    result = validate_pdf(str(fake), "R94")
    assert result.outcome == ValidationOutcome.REJECTED
    assert "Magic bytes" in result.reason


def test_undersized_placeholder_rejected(tmp_path):
    tiny = tmp_path / "tiny.pdf"
    tiny.write_bytes(b"%PDF-1.4\nplaceholder")
    result = validate_pdf(str(tiny), "R94")
    assert result.outcome == ValidationOutcome.REJECTED


def test_valid_regulation_pdf_accepted(tmp_path):
    pdf = tmp_path / "r94.pdf"
    _write_min_pdf(
        pdf,
        "UN REGULATION No. 94\nFrontal collision protection.\n" + ("Requirement text. " * 40),
    )
    result = validate_pdf(str(pdf), "R94")
    assert result.outcome == ValidationOutcome.ACCEPTED


def test_duplicate_skipped(db_session, tmp_path):
    tracker = ChangeTracker()
    text = "UN REGULATION No. 95\nLateral collision.\n" + ("Body " * 50)
    pdf = tmp_path / "r95.pdf"
    _write_min_pdf(pdf, text)
    url = "https://unece.org/example/r95.pdf"

    status1, text_hash, _ = tracker.classify(
        db_session,
        source_url=url,
        authority="UNECE",
        regulation_id="R95",
        text=text,
        file_path=str(pdf),
    )
    assert status1 == "NEW"
    tracker.upsert(
        db_session,
        source_url=url,
        authority="UNECE",
        regulation_id="R95",
        text_hash=text_hash,
        file_path=str(pdf),
    )
    status2, _, _ = tracker.classify(
        db_session,
        source_url=url,
        authority="UNECE",
        regulation_id="R95",
        text=text,
        file_path=str(pdf),
    )
    assert status2 == "DUPLICATE"


def test_changed_doc_detected(db_session):
    tracker = ChangeTracker()
    url = "https://unece.org/example/r94.pdf"
    tracker.upsert(
        db_session,
        source_url=url,
        authority="UNECE",
        regulation_id="R94",
        text_hash="hash_v1",
        file_path="/storage/UNECE/old.pdf",
    )
    status, new_hash, _ = tracker.classify(
        db_session,
        source_url=url,
        authority="UNECE",
        regulation_id="R94",
        text="UN REGULATION No. 94\nRevised requirement wording.",
        file_path="/storage/UNECE/new.pdf",
    )
    assert status == "CHANGED"
    assert new_hash != "hash_v1"


def test_off_list_domain_refused():
    with pytest.raises(DomainNotAllowedError):
        assert_url_allowed("https://evil.example.com/doc.pdf", "UNECE")


def test_remote_unchanged_when_etag_matches():
    assert remote_changed({"etag": '"abc"'}, {"etag": '"abc"'}) is False
    assert remote_changed({"etag": '"abc"'}, {"etag": '"def"'}) is True
