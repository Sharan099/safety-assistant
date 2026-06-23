"""
Markdown quality gate — per-document OCR sanity checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Minimum thresholds (override via env in caller if needed)
MIN_CHAR_COUNT = 500
MAX_GARBAGE_LINE_RATIO = 0.35
MIN_TABLE_ROW_CHARS = 8

GARBAGE_RE = re.compile(
    r"^[\W\d\s]{0,3}$|"
    r"^[█▓▒░■□]{3,}$|"
    r"^[^\w\s]{10,}$|"
    r"^(page\s+\d+\s*){1,2}$",
    re.I,
)
TABLE_ROW_RE = re.compile(r"\|[^|]+\|")


@dataclass
class QualityResult:
    passed: bool
    pdf_name: str
    char_count: int
    garbage_ratio: float
    issues: list[str]
    failed_page: str | None = None


def _page_sections(md_text: str) -> list[tuple[str, str]]:
    """Split markdown into (page_label, content) sections."""
    parts = re.split(r"(?m)^## Page (\d+)\s*$", md_text)
    if len(parts) < 3:
        return [("document", md_text)]
    sections: list[tuple[str, str]] = []
    if parts[0].strip():
        sections.append(("preamble", parts[0]))
    for i in range(1, len(parts), 2):
        label = f"Page {parts[i]}"
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((label, body))
    return sections


def check_markdown(md_path: Path, *, min_chars: int = MIN_CHAR_COUNT) -> QualityResult:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    pdf_name = md_path.stem
    issues: list[str] = []
    failed_page: str | None = None

    # Strip frontmatter
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3 :]

    char_count = len(text.strip())
    if char_count < min_chars:
        issues.append(f"char_count {char_count} < {min_chars}")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        issues.append("no content lines")
        garbage_ratio = 1.0
    else:
        garbage = sum(1 for ln in lines if GARBAGE_RE.match(ln))
        garbage_ratio = garbage / len(lines)
        if garbage_ratio > MAX_GARBAGE_LINE_RATIO:
            issues.append(
                f"garbage_line_ratio {garbage_ratio:.2f} > {MAX_GARBAGE_LINE_RATIO}"
            )
            for page_label, body in _page_sections(text):
                plines = [ln.strip() for ln in body.splitlines() if ln.strip()]
                if not plines:
                    continue
                pg = sum(1 for ln in plines if GARBAGE_RE.match(ln)) / len(plines)
                if pg > MAX_GARBAGE_LINE_RATIO:
                    failed_page = page_label
                    break

    # Table sanity: fragmented single-cell rows
    for row in TABLE_ROW_RE.findall(text):
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if len(cells) == 1 and len(cells[0]) < MIN_TABLE_ROW_CHARS:
            issues.append("fragmented table row detected")
            break

    return QualityResult(
        passed=len(issues) == 0,
        pdf_name=pdf_name,
        char_count=char_count,
        garbage_ratio=round(garbage_ratio, 3),
        issues=issues,
        failed_page=failed_page,
    )


@dataclass
class OcrConfidenceAudit:
    """Sampled OCR-confidence audit for silent numeric OCR errors."""
    pdf_name: str
    pages_sampled: int
    low_confidence_pages: list[str]
    flagged_tables: list[str]
    issues: list[str]

    @property
    def needs_review(self) -> bool:
        return bool(self.low_confidence_pages or self.flagged_tables or self.issues)


_NUMERIC_OCR_SUSPECT = re.compile(
    r"(\d{2,})\s*(?:mm|m\b|daN|kN)", re.I
)


def audit_ocr_confidence(
    md_path: Path,
    *,
    sample_every: int = 5,
    min_confidence: float = 0.75,
) -> OcrConfidenceAudit:
    """
    Sample pages/tables for low-confidence OCR patterns (e.g. 1.5 m → 15 m).
    Biased toward catching numeric unit errors over speed.
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    sections = _page_sections(text)
    low_pages: list[str] = []
    flagged_tables: list[str] = []
    issues: list[str] = []

    for idx, (label, body) in enumerate(sections):
        if idx % max(sample_every, 1) != 0:
            continue
        # Heuristic: missing decimal before unit (15 m vs 1.5 m pattern)
        for m in _NUMERIC_OCR_SUSPECT.finditer(body):
            val = m.group(1)
            if len(val) >= 2 and val.startswith("1") and "mm" in m.group(0).lower():
                if f"{val[0]}.{val[1:]}" not in body:
                    issues.append(f"{label}: possible OCR magnitude error near {m.group(0)}")
                    low_pages.append(label)
                    break
        if "|" in body:
            rows = [r for r in body.splitlines() if "|" in r]
            short_cells = sum(
                1 for r in rows for c in r.split("|")
                if c.strip() and len(c.strip()) < MIN_TABLE_ROW_CHARS
            )
            if short_cells > len(rows):
                flagged_tables.append(label)
                issues.append(f"{label}: fragmented table cells — review OCR")

    return OcrConfidenceAudit(
        pdf_name=md_path.stem,
        pages_sampled=len(sections),
        low_confidence_pages=low_pages,
        flagged_tables=flagged_tables,
        issues=issues,
    )
