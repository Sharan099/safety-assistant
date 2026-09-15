"""Untrusted-upload boundaries: clamd INSTREAM client against a fake daemon; OCR adapter on an
image-only page; the parser stays honest (NEEDS_REVIEW) when OCR is absent."""

from __future__ import annotations

import socket
import struct
import threading
from collections.abc import Iterator

import pymupdf
import pytest

from safety_assistant.ingestion.parse.ocr import NoOcr, OcrUnavailable, ocr_from_settings
from safety_assistant.ingestion.parse.pymupdf_parser import PyMuPDFParser
from safety_assistant.ingestion.validation.scan import (
    ClamdScanner,
    NoScanner,
    ScannerUnavailable,
    scanner_from_settings,
)


@pytest.fixture
def fake_clamd() -> Iterator[tuple[int, list[bytes]]]:
    """Minimal clamd: reads INSTREAM chunks, replies FOUND when the payload contains the EICAR marker."""
    received: list[bytes] = []
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve() -> None:
        conn, _ = srv.accept()
        with conn:
            buf = b""
            while b"\0" not in buf:
                buf += conn.recv(64)
            assert buf.startswith(b"zINSTREAM\0")
            payload = b""
            while True:
                header = b""
                while len(header) < 4:
                    header += conn.recv(4 - len(header))
                (n,) = struct.unpack("!I", header)
                if n == 0:
                    break
                chunk = b""
                while len(chunk) < n:
                    chunk += conn.recv(n - len(chunk))
                payload += chunk
            received.append(payload)
            conn.sendall(b"stream: Eicar-Test-Signature FOUND\0" if b"EICAR" in payload else b"stream: OK\0")

    threading.Thread(target=serve, daemon=True).start()
    yield port, received
    srv.close()


def test_clamd_client_streams_bytes_and_reads_verdict(fake_clamd: tuple[int, list[bytes]]) -> None:
    port, received = fake_clamd
    data = b"%PDF-1.4 " + b"x" * 3000
    verdict = ClamdScanner("127.0.0.1", port, chunk=1024).scan(data)
    assert verdict.clean and received == [data]


def test_clamd_client_reports_infected(fake_clamd: tuple[int, list[bytes]]) -> None:
    port, _ = fake_clamd
    verdict = ClamdScanner("127.0.0.1", port).scan(b"%PDF EICAR test body")
    assert verdict.clean is False and "Eicar" in (verdict.detail or "")


def test_unreachable_scanner_is_an_error_not_a_pass() -> None:
    with pytest.raises(ScannerUnavailable):
        ClamdScanner("127.0.0.1", 1, timeout_s=0.5).scan(b"x")
    assert isinstance(scanner_from_settings("none", host="h", port=1), NoScanner)
    assert isinstance(scanner_from_settings("clamav", host="h", port=1), ClamdScanner)


def _image_only_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    # draw a shape so the page is not blank, but add no text layer
    page.draw_rect(pymupdf.Rect(10, 10, 190, 90), color=(0, 0, 0), width=2)
    return doc.tobytes()


class _FakeOcr:
    name = "fake"

    def ocr_png(self, png: bytes) -> str:
        assert png.startswith(b"\x89PNG")
        return "The head performance criterion shall not exceed 1000. " * 8


def test_parser_without_ocr_flags_scanned_pages_and_with_ocr_recovers_text() -> None:
    data = _image_only_pdf()
    without = PyMuPDFParser(ocr=None).parse(data, source_sha256="x")
    assert without.pages[0].needs_ocr and without.report is not None and without.report.status == "NEEDS_REVIEW"
    with_ocr = PyMuPDFParser(ocr=_FakeOcr()).parse(data, source_sha256="x")
    assert not with_ocr.pages[0].needs_ocr and "head performance" in with_ocr.pages[0].text
    assert PyMuPDFParser(ocr=_FakeOcr()).config_hash() != PyMuPDFParser().config_hash()


def test_ocr_adapters() -> None:
    with pytest.raises(OcrUnavailable):
        NoOcr().ocr_png(b"\x89PNG")
    assert ocr_from_settings("none").name == "none"


def test_missing_ocr_binary_degrades_to_no_ocr(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.config import get_settings
    from safety_assistant.ingestion.workflows.ingest import _default_parser

    monkeypatch.setenv("OCR_PROVIDER", "tesseract")
    monkeypatch.setattr("shutil.which", lambda _b: None)
    get_settings.cache_clear()
    try:
        parser = _default_parser(get_settings())
    finally:
        get_settings.cache_clear()
    assert parser.ocr is None  # the upload proceeds; scanned pages stay flagged
