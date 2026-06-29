"""Split page-flattened regulation text into clause-aligned sections."""

from __future__ import annotations

import re

# UNECE style: "5.2.1.4." on its own line, or "5.2.1.4. The Thorax..."
_CLAUSE_ONLY = re.compile(r"^(\d+(?:\.\d+)+\.?)\s*$")
_CLAUSE_WITH_TEXT = re.compile(r"^(\d+(?:\.\d+)+\.?)\s+(.+)$")
_PAGE_TITLE = re.compile(r"^page\s+\d+", re.I)


def is_page_section_title(title: str) -> bool:
    return bool(_PAGE_TITLE.match((title or "").strip()))


def split_body_by_clauses(
    body: str,
    *,
    page_number: int | None = None,
) -> list[tuple[str | None, str, str, int | None]]:
    """
    Return (clause_number, section_title, body_text, page_number) blocks.

    Handles UNECE numbering where clause numbers often sit on their own line.
    """
    lines = body.splitlines()
    blocks: list[tuple[str | None, str, list[str], int | None]] = []
    current_clause: str | None = None
    current_title = ""
    current_lines: list[str] = []
    current_page = page_number

    def flush() -> None:
        nonlocal current_clause, current_title, current_lines, current_page
        text = "\n".join(current_lines).strip()
        if text or current_clause:
            blocks.append((current_clause, current_title, current_lines[:], current_page))
        current_clause = None
        current_title = ""
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("<!--"):
            page_m = re.match(r"<!--\s*page:\s*(\d+)\s*-->", stripped, re.I)
            if page_m:
                current_page = int(page_m.group(1))
            continue

        cm = _CLAUSE_ONLY.match(stripped)
        if cm:
            flush()
            current_clause = cm.group(1).rstrip(".")
            current_title = f"§{current_clause}"
            continue

        cwt = _CLAUSE_WITH_TEXT.match(stripped)
        if cwt:
            flush()
            current_clause = cwt.group(1).rstrip(".")
            rest = cwt.group(2).strip()
            current_title = f"§{current_clause} {rest[:80]}".strip()
            current_lines.append(rest)
            continue

        current_lines.append(line)

    flush()

    if len(blocks) <= 1:
        return [(None, "", body.strip(), page_number)] if body.strip() else []

    out: list[tuple[str | None, str, str, int | None]] = []
    for clause, title, line_buf, pg in blocks:
        text = "\n".join(line_buf).strip()
        if not text and not clause:
            continue
        out.append((clause, title or (f"§{clause}" if clause else ""), text, pg))
    return out or [(None, "", body.strip(), page_number)]


def _is_direct_subclause(parent: str, child: str) -> bool:
    """True when child is exactly one numbering level below parent (e.g. 2.12.4 → 2.12.4.1)."""
    if not parent or not child:
        return False
    return child.startswith(parent + ".") and child.count(".") == parent.count(".") + 1


def merge_subclause_blocks(
    blocks: list[tuple[str | None, str, str, int | None]],
) -> list[tuple[str | None, str, str, int | None]]:
    """
    Merge immediate sub-clauses into their parent block so definitions like
    2.12.4 (ELR) stay with qualifying conditions 2.12.4.1 / 2.12.4.2.
    """
    if len(blocks) <= 1:
        return blocks

    merged: list[tuple[str | None, str, str, int | None]] = []
    i = 0
    while i < len(blocks):
        clause, title, body, page = blocks[i]
        if clause:
            parts = [body] if body.strip() else []
            j = i + 1
            parent_ends_colon = body.rstrip().endswith(":")
            while j < len(blocks):
                child_clause, _child_title, child_body, _child_page = blocks[j]
                if (
                    child_clause
                    and _is_direct_subclause(clause, child_clause)
                    and parent_ends_colon
                ):
                    parts.append(f"{child_clause}.\n{child_body}".strip())
                    j += 1
                else:
                    break
            merged.append((clause, title, "\n".join(parts).strip(), page))
            i = j
        else:
            merged.append(blocks[i])
            i += 1
    return merged


def merge_trailing_subclauses(
    blocks: list[tuple[str | None, str, str, int | None]],
) -> list[tuple[str | None, str, str, int | None]]:
    """
    Merge numbered sub-clauses into a block whose body ends with ':' (common when
    PDF text keeps 2.12.4 on a section chunk but splits 2.12.4.1 into its own block).
    """
    if len(blocks) <= 1:
        return blocks

    out: list[tuple[str | None, str, str, int | None]] = []
    i = 0
    while i < len(blocks):
        clause, title, body, page = blocks[i]
        stripped = body.rstrip()
        if stripped.endswith(":"):
            j = i + 1
            parts = [body]
            family_prefix: str | None = None
            while j < len(blocks):
                child_clause, _child_title, child_body, _child_page = blocks[j]
                if not child_clause or not child_body.strip():
                    break
                if family_prefix is None:
                    family_prefix = (
                        child_clause.rsplit(".", 1)[0] if "." in child_clause else child_clause
                    )
                elif not child_clause.startswith(family_prefix + "."):
                    break
                parts.append(f"{child_clause}.\n{child_body}".strip())
                j += 1
            if j > i + 1:
                out.append((clause, title, "\n".join(parts).strip(), page))
                i = j
                continue
        out.append(blocks[i])
        i += 1
    return out
