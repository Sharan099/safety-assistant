"""Load PDF, Markdown, and plain-text documents for ingestion."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz
from loguru import logger

# pdfplumber is preferred for table structure; fall back to PyMuPDF text-only.
try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore


def load_pages(path: Path) -> list[dict[str, Any]]:
    """Return page-like dicts: {page_number, text}. OCR hook for scanned PDFs later."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in (".md", ".txt"):
        return _load_text(path)
    raise ValueError(f"Unsupported format: {suffix}")


def _table_to_markdown(table: list[list[Any]]) -> str:
    """Convert a pdfplumber table (list of rows) to a compact markdown table."""
    if not table:
        return ""
    rows: list[list[str]] = []
    width = max((len(r) for r in table if r), default=0)
    if width == 0:
        return ""
    for raw in table:
        raw = raw or []
        cells = [
            re.sub(r"\s+", " ", (c or "").replace("\n", " ")).strip()
            for c in list(raw) + [""] * (width - len(raw))
        ]
        rows.append(cells[:width])
    # Drop fully empty rows
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    # Drop empty columns
    keep = [i for i in range(width) if any(r[i] for r in rows)]
    if not keep:
        return ""
    rows = [[r[i] for i in keep] for r in rows]
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _load_pdf_pdfplumber(path: Path) -> list[dict[str, Any]] | None:
    """Extract text + tables as markdown blocks per page via pdfplumber."""
    if pdfplumber is None:
        return None
    pages: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                table_blocks: list[str] = []
                for ti, table in enumerate(page.extract_tables() or []):
                    md = _table_to_markdown(table)
                    if not md or md.count("|") < 4:
                        continue
                    # Skip near-empty / junk detections (1x2 empty)
                    if md.count("\n") < 2:
                        continue
                    table_blocks.append(f"\n\n[Table {ti + 1} on page {i + 1}]\n{md}\n")
                if table_blocks:
                    # Append structured tables after flowing text so both are retained.
                    text = (text + "".join(table_blocks)).strip()
                if not text:
                    logger.debug(
                        "Page {} in {} has no extractable text (OCR candidate)",
                        i + 1,
                        path.name,
                    )
                pages.append(
                    {
                        "page_number": i + 1,
                        "text": text,
                        "n_tables": len(table_blocks),
                    }
                )
        return pages
    except Exception as exc:
        logger.warning("pdfplumber failed for {} ({}): falling back to PyMuPDF", path.name, exc)
        return None


def _load_pdf_fitz(path: Path) -> list[dict[str, Any]]:
    doc = fitz.open(path)
    pages: list[dict[str, Any]] = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        if not text.strip():
            logger.debug("Page {} in {} has no extractable text (OCR candidate)", i + 1, path.name)
        pages.append({"page_number": i + 1, "text": text.strip(), "n_tables": 0})
    doc.close()
    return pages


def _load_pdf(path: Path) -> list[dict[str, Any]]:
    """Prefer pdfplumber (tables → markdown); fall back to PyMuPDF text."""
    pages = _load_pdf_pdfplumber(path)
    if pages is not None:
        n_tables = sum(int(p.get("n_tables") or 0) for p in pages)
        logger.info(
            "Loaded {} via pdfplumber ({} pages, {} table block(s))",
            path.name,
            len(pages),
            n_tables,
        )
        return pages
    logger.info("Loaded {} via PyMuPDF text-only", path.name)
    return _load_pdf_fitz(path)


def _load_text(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [{"page_number": 1, "text": text, "n_tables": 0}] if text else []


def document_type_for(path: Path) -> str:
    return path.suffix.lstrip(".").upper() or "UNKNOWN"
