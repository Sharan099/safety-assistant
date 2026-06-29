"""Extract annex/section structure from flattened regulation page text."""

from __future__ import annotations

import re
from typing import Any

_ANNEX_HEADING = re.compile(r"^(Annex\s+\d+(?:\s*[-–]\s*Appendix\s+\d+)?)\s*$", re.I | re.M)
_SECTION_HEADING = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+([A-Z].{3,})$", re.M)
_CLAUSE_ONLY = re.compile(r"^(\d+(?:\.\d+)+\.)\s*$")
_CLAUSE_WITH_TITLE = re.compile(r"^(\d+(?:\.\d+)+\.)\s+(.+)$")


def _clause_boundary_id(stripped: str) -> str | None:
    """UNECE clause number at line start (OCR base PDFs often bury these inside annex blocks)."""
    m = _CLAUSE_ONLY.match(stripped)
    if m:
        return m.group(1).rstrip(".")
    m = _CLAUSE_WITH_TITLE.match(stripped)
    if m and re.match(r"[A-Za-z]", m.group(2)):
        return m.group(1).rstrip(".")
    return None


def _is_new_structural_branch(current_id: str, clause_id: str) -> bool:
    """True when a clause line starts a different major branch (e.g. Annex 3 → 7.6.2, 6.2.5 → 7.6.2)."""
    if current_id.startswith("Annex"):
        return True
    cur_top = current_id.split(".")[0].split("_")[0]
    cl_top = clause_id.split(".")[0]
    if cur_top.isdigit() and cl_top.isdigit() and cur_top != cl_top:
        return True
    if current_id != clause_id and not clause_id.startswith(current_id + "."):
        # Sibling branch under same chapter (6.2.5 vs 7.6.2) — promote boundary
        if cur_top.isdigit() and cl_top.isdigit() and cur_top == cl_top:
            cur_depth = current_id.count(".")
            cl_depth = clause_id.count(".")
            if cl_depth <= cur_depth and not clause_id.startswith(current_id):
                return True
    return False


def pages_to_structured_blocks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert parsed PDF pages into logical blocks with heading_path and page span.

    Each block: {heading_path, section_id, block_type, text, page_start, page_end, tables}
    """
    combined: list[str] = []
    for page in pages:
        pn = page["page_number"]
        combined.append(f"<!-- page: {pn} -->")
        combined.append(page.get("text", ""))
        for table_md in page.get("tables") or []:
            combined.append("<!-- chunk_type:table -->")
            combined.append(table_md)
    full_text = "\n".join(combined)

    blocks: list[dict[str, Any]] = []
    current_path = "Document"
    current_id = "General"
    current_type = "section"
    buf: list[str] = []
    page_start: int | None = None
    page_end: int | None = None

    def flush() -> None:
        nonlocal buf, page_start, page_end
        text = "\n".join(buf).strip()
        if text:
            blocks.append(
                {
                    "heading_path": current_path,
                    "section_id": current_id,
                    "block_type": current_type,
                    "text": text,
                    "page_start": page_start,
                    "page_end": page_end,
                }
            )
        buf = []
        page_start = None
        page_end = None

    for line in full_text.splitlines():
        page_m = re.match(r"<!--\s*page:\s*(\d+)\s*-->", line.strip(), re.I)
        if page_m:
            page_end = int(page_m.group(1))
            if page_start is None:
                page_start = page_end
            buf.append(line)
            continue

        stripped = line.strip()
        annex_m = _ANNEX_HEADING.match(stripped)
        if annex_m:
            flush()
            annex = annex_m.group(1).strip()
            current_path = annex
            current_id = annex.replace(" ", "_")
            current_type = "annex"
            continue

        sec_m = _SECTION_HEADING.match(stripped)
        if sec_m and len(sec_m.group(1).split(".")) <= 2:
            flush()
            sec_id = sec_m.group(1).rstrip(".")
            title = sec_m.group(2).strip()
            current_id = sec_id
            current_path = f"{current_path} > §{sec_id} {title[:60]}"
            current_type = "section"
            buf.append(line)
            continue

        clause_id = _clause_boundary_id(stripped)
        if clause_id and (
            current_id in ("General", "Document")
            or _is_new_structural_branch(current_id, clause_id)
        ):
            flush()
            current_id = clause_id
            current_path = f"Document > §{clause_id}"
            current_type = "clause"
            buf.append(line)
            continue

        buf.append(line)

    flush()
    return blocks or [
        {
            "heading_path": "Document",
            "section_id": "General",
            "block_type": "section",
            "text": full_text,
            "page_start": pages[0]["page_number"] if pages else 1,
            "page_end": pages[-1]["page_number"] if pages else 1,
        }
    ]
