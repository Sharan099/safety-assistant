"""Upload/ingestion validation boundary."""

from __future__ import annotations

import pytest


def test_password_protected_pdf_is_refused() -> None:
    import pymupdf

    from safety_assistant.ingestion.validation import ValidationError, validate_pdf_bytes

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(ValidationError, match="password"):
        validate_pdf_bytes(data, expected_sha256=None, max_bytes=10_000_000, max_pages=10)
