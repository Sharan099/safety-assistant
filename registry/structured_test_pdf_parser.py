"""Deterministic structured test-report PDF → harness JSON (no LLM)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from parser.pdf_parser import PDFParser

_TEST_ID_RE = re.compile(r"\b(TEST-\d{4}-\d{2,3}-\d{3})\b", re.I)
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_PROGRAM_RE = re.compile(r"Program:\s*([^\n\r]+)", re.I)
_IMPACT_RE = re.compile(r"Impact\s*mode:\s*([A-Z_]+)", re.I)
_DUMMY_RE = re.compile(r"Dummy:\s*([^\n\r]+)", re.I)
_INJURY_ROW_RE = re.compile(
    r"\b(ThCC|HIC36?|TCFC)\b\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*(mm|g|kN)?",
    re.I,
)


def _extract_full_text(pdf_path: Path) -> str:
    parser = PDFParser(str(pdf_path))
    pages = parser.parse(extract_tables=False)
    return "\n".join(p.get("text", "") or "" for p in pages)


def parse_structured_test_pdf(
    pdf_path: Path,
    *,
    linked_regulation_clause: str,
    default_test_type: str = "PHYSICAL_CRASH",
    default_impact_mode: str = "FRONTAL_OFFSET",
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Parse lab-style PDF into harness ingest records.
    Returns (records, errors). errors non-empty => do not ingest.
    """
    errors: list[str] = []
    if not pdf_path.exists():
        return [], [f"PDF not found: {pdf_path}"]

    text = _extract_full_text(pdf_path)
    if not text.strip():
        return [], ["PDF contains no extractable text"]

    test_id_m = _TEST_ID_RE.search(text)
    date_m = _DATE_RE.search(text)
    if not test_id_m:
        errors.append("Could not find test_id pattern TEST-YYYY-XX-NNN in PDF")
    if not date_m:
        errors.append("Could not find date YYYY-MM-DD in PDF")

    injury_rows: list[dict[str, Any]] = []
    for m in _INJURY_ROW_RE.finditer(text):
        crit_raw = m.group(1).upper()
        if crit_raw in ("HIC", "HIC36"):
            crit = "HIC36"
        elif crit_raw == "THCC":
            crit = "ThCC"
        else:
            crit = crit_raw
        injury_rows.append(
            {
                "channel": "PARSED",
                "injury_criterion": crit,
                "value": float(m.group(2)),
                "linked_regulation_clause": linked_regulation_clause,
            }
        )

    if not injury_rows:
        errors.append("No injury criterion rows (ThCC/HIC/TCFC) found in PDF text")

    if errors:
        return [], errors

    assert test_id_m is not None and date_m is not None
    try:
        datetime.strptime(date_m.group(1), "%Y-%m-%d")
    except ValueError:
        return [], [f"Invalid date in PDF: {date_m.group(1)}"]

    program_m = _PROGRAM_RE.search(text)
    impact_m = _IMPACT_RE.search(text)
    dummy_m = _DUMMY_RE.search(text)

    record: dict[str, Any] = {
        "test_id": test_id_m.group(1).upper(),
        "program": (program_m.group(1).strip() if program_m else "UPLOADED"),
        "date": date_m.group(1),
        "test_type": default_test_type,
        "impact_mode": (impact_m.group(1).upper() if impact_m else default_impact_mode),
        "dummy": dummy_m.group(1).strip() if dummy_m else None,
        "confidential_tier": True,
        "results": injury_rows,
    }
    return [record], []
