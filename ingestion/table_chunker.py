"""Detect and extract atomic markdown tables from regulation section bodies."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

TABLE_MARKER_RE = re.compile(r"<!--\s*chunk_type:table\s*-->", re.I)
PIPE_TABLE_ROW = re.compile(r"^\|.+\|$")


@dataclass
class TableBlock:
    table_id: str
    markdown: str
    summary: str
    preamble: str
    postamble: str


def _table_id(clause_number: str | None, section_title: str, body: str) -> str:
    key = f"{clause_number or ''}|{section_title}|{body[:120]}"
    return "tbl_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def _summarize_table(markdown: str) -> str:
    lines = [ln for ln in markdown.splitlines() if ln.strip() and not ln.strip().startswith("| ---")]
    header = next((ln for ln in lines if ln.startswith("|")), "")
    cells = [c.strip() for c in header.strip("|").split("|") if c.strip()]
    if cells:
        return f"Table: {', '.join(cells[:4])}" + ("…" if len(cells) > 4 else "")
    return "Structured compliance table"


def _is_pipe_table_block(lines: list[str], start: int) -> int | None:
    """Return end index (exclusive) if lines[start:] begin a pipe table."""
    if start >= len(lines) or not PIPE_TABLE_ROW.match(lines[start].strip()):
        return None
    end = start
    while end < len(lines) and PIPE_TABLE_ROW.match(lines[end].strip()):
        end += 1
    if end - start >= 2:
        return end
    return None


def extract_tables_from_body(
    body: str,
    *,
    clause_number: str | None = None,
    section_title: str = "",
    add_summary: bool = True,
) -> tuple[str, list[TableBlock]]:
    """
    Pull atomic table blocks out of section body.

    Returns (remainder_text, table_blocks).
    """
    if not body.strip():
        return body, []

    lines = body.splitlines()
    remainder: list[str] = []
    tables: list[TableBlock] = []
    i = 0
    preamble_buf: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if TABLE_MARKER_RE.search(stripped):
            preamble = "\n".join(preamble_buf).strip()
            preamble_buf = []
            i += 1
            table_lines: list[str] = []
            while i < len(lines):
                if TABLE_MARKER_RE.search(lines[i]) or (
                    lines[i].strip().startswith("<!-- chunk_type:")
                    and "table" not in lines[i].lower()
                ):
                    break
                if lines[i].strip():
                    table_lines.append(lines[i])
                i += 1
                if table_lines and not PIPE_TABLE_ROW.match(table_lines[-1].strip()):
                    if len(table_lines) >= 2 and any(
                        PIPE_TABLE_ROW.match(tl.strip()) for tl in table_lines
                    ):
                        break
            md = "\n".join(table_lines).strip()
            if md:
                tid = _table_id(clause_number, section_title, md)
                summary = _summarize_table(md) if add_summary else ""
                tables.append(TableBlock(tid, md, summary, preamble, ""))
            continue

        end = _is_pipe_table_block(lines, i)
        if end is not None:
            preamble = "\n".join(preamble_buf).strip()
            preamble_buf = []
            md = "\n".join(lines[i:end]).strip()
            tid = _table_id(clause_number, section_title, md)
            summary = _summarize_table(md) if add_summary else ""
            tables.append(TableBlock(tid, md, summary, preamble, ""))
            i = end
            continue

        preamble_buf.append(line)
        i += 1

    remainder_text = "\n".join(preamble_buf).strip()
    return remainder_text, tables
