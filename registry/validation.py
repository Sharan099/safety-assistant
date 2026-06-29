"""PDF validation gate (FR-5…FR-8)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum

import fitz
from loguru import logger

from parser.pdf_parser import PDFParser

PDF_MAGIC = b"%PDF-"
MIN_PDF_BYTES = int(os.getenv("MIN_PDF_BYTES", "2048"))
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(80 * 1024 * 1024)))

PLACEHOLDER_BYTE_PATTERNS = [
    b"git-lfs.github.com",
    b"<html",
    b"<!doctype",
    b"placeholder",
    b"lorem ipsum",
]

IMAGE_ONLY_MIN_CHARS_PER_PAGE = int(os.getenv("IMAGE_ONLY_MIN_CHARS_PER_PAGE", "80"))


class ValidationOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_OCR = "needs_ocr"


@dataclass
class ValidationResult:
    outcome: ValidationOutcome
    reason: str
    page_count: int = 0
    extracted_text: str = ""


def _regulation_id_in_text(text: str, regulation_id: str) -> bool:
    if not regulation_id:
        return True
    upper = text.upper()
    rid = regulation_id.upper().strip()
    patterns = [rid, rid.replace(" ", ""), rid.replace("UN ", "")]
    if rid.startswith("R") and rid[1:].isdigit():
        num = rid[1:]
        patterns.extend([f"REGULATION NO. {num}", f"REGULATION NO {num}", f"UN {rid}"])
    return any(p and p in upper for p in patterns)


def validate_pdf(file_path: str, expected_regulation_id: str) -> ValidationResult:
    """Ordered validation gate: magic → open/pages → size → reg id → placeholder/OCR."""
    try:
        with open(file_path, "rb") as fh:
            header = fh.read(512)
    except OSError as exc:
        return ValidationResult(ValidationOutcome.REJECTED, f"Cannot read file: {exc}")

    if not header.startswith(PDF_MAGIC):
        return ValidationResult(ValidationOutcome.REJECTED, "Magic bytes check failed: not %PDF-")

    for pattern in PLACEHOLDER_BYTE_PATTERNS:
        if pattern in header.lower():
            return ValidationResult(ValidationOutcome.REJECTED, f"Placeholder pattern detected: {pattern!r}")

    size = os.path.getsize(file_path)
    if size < MIN_PDF_BYTES:
        return ValidationResult(ValidationOutcome.REJECTED, f"File too small ({size} < {MIN_PDF_BYTES})")
    if size > MAX_PDF_BYTES:
        return ValidationResult(ValidationOutcome.REJECTED, f"File too large ({size} > {MAX_PDF_BYTES})")

    try:
        with fitz.open(file_path) as doc:
            page_count = len(doc)
            if page_count <= 0:
                return ValidationResult(ValidationOutcome.REJECTED, "PDF has zero pages")
    except Exception as exc:
        return ValidationResult(ValidationOutcome.REJECTED, f"PDF library open failed: {exc}")

    try:
        parser = PDFParser(file_path)
        pages = parser.parse(extract_tables=False)
        extracted = "\n".join(p.get("text", "") for p in pages)
    except Exception as exc:
        return ValidationResult(ValidationOutcome.REJECTED, f"Text extraction failed: {exc}")

    stripped = re.sub(r"\s+", "", extracted)
    if len(stripped) < 50:
        return ValidationResult(
            ValidationOutcome.REJECTED,
            "Synthetic/empty document: insufficient text layer",
            page_count=page_count,
            extracted_text=extracted,
        )

    chars_per_page = len(stripped) / max(page_count, 1)
    if chars_per_page < IMAGE_ONLY_MIN_CHARS_PER_PAGE:
        return ValidationResult(
            ValidationOutcome.NEEDS_OCR,
            f"Image-only scan suspected ({chars_per_page:.0f} chars/page)",
            page_count=page_count,
            extracted_text=extracted,
        )

    if not _regulation_id_in_text(extracted, expected_regulation_id):
        return ValidationResult(
            ValidationOutcome.REJECTED,
            f"Expected regulation id {expected_regulation_id!r} not found in text",
            page_count=page_count,
            extracted_text=extracted,
        )

    logger.info(f"Validation accepted: {file_path} ({page_count} pages)")
    return ValidationResult(
        ValidationOutcome.ACCEPTED,
        "ok",
        page_count=page_count,
        extracted_text=extracted,
    )
