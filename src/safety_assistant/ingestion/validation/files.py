"""Source-file validation gate (CLAUDE.md §14): registry hash match, size
limit, PDF magic bytes, page-count limit. Runs before any parser sees bytes."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from safety_assistant.ingestion.fetch.blobstore import sha256_bytes

PDF_MAGIC = b"%PDF-"


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedFile:
    sha256: str
    size_bytes: int
    media_type: str
    page_count: int


def validate_pdf_bytes(
    data: bytes, *, expected_sha256: str | None, max_bytes: int, max_pages: int, expected_size: int | None = None
) -> ValidatedFile:
    if len(data) == 0:
        raise ValidationError("empty file")
    if len(data) > max_bytes:
        raise ValidationError(f"file too large: {len(data)} > {max_bytes} bytes")
    if expected_size is not None and len(data) != expected_size:
        raise ValidationError(f"size mismatch: registry={expected_size} actual={len(data)}")
    if not data.startswith(PDF_MAGIC):
        raise ValidationError("not a PDF (magic bytes)")
    digest = sha256_bytes(data)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValidationError(f"sha256 mismatch: registry={expected_sha256} actual={digest}")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
            page_count = len(doc)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"PDF cannot be opened: {exc}") from exc
    if page_count == 0:
        raise ValidationError("PDF has no pages")
    if page_count > max_pages:
        raise ValidationError(f"too many pages: {page_count} > {max_pages}")
    return ValidatedFile(sha256=digest, size_bytes=len(data), media_type="application/pdf", page_count=page_count)
