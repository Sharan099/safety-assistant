"""Event-based chunking for synthetic internal markdown documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_EVENT_SECTIONS = (
    "Test identification",
    "Configuration",
    "Measurements",
    "Regulatory assessment",
    "Observation",
    "Correlation summary",
    "Delta table",
    "Problem statement",
    "Causal chain",
    "Requirement traceability",
    "Executive overview",
    "Key metrics",
)

_TABLE_RE = re.compile(r"^\|.+\|", re.M)


def is_synthetic_markdown(md_path: Path, meta: dict) -> bool:
    if meta.get("is_synthetic", "").lower() == "true":
        return True
    return "synthetic" in str(md_path).lower() or meta.get("source_type") == "synthetic"


def chunk_synthetic_events(
    md_path: Path,
    text: str,
    meta: dict,
    *,
    make_chunk,
    regulation: str,
    file_slug: str,
    pdf_name: str,
) -> list[dict]:
    """Bundle observation + measurement + pass/fail judgement into atomic event chunks."""
    chunks: list[dict] = []
    sections = re.split(r"\n##\s+", text)
    sec_idx = 0
    for block in sections[1:]:
        sec_idx += 1
        lines = block.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue
        heading_path = f"{regulation} > {title}"
        is_table = bool(_TABLE_RE.search(body))
        chunk_type = "table" if is_table else "event"
        if title in _EVENT_SECTIONS or is_table:
            event_text = (
                f"{regulation} | {meta.get('revision', 'synthetic')} | {title}\n\n"
                f"{body}"
            )
            chunks.append(
                make_chunk(
                    regulation=regulation,
                    file_slug=file_slug,
                    pdf_name=pdf_name,
                    markdown_file=md_path.name,
                    chunk_type=chunk_type,
                    parent_id=None,
                    heading_path=heading_path,
                    section_title=title,
                    section_level=2,
                    text=event_text,
                    seq=sec_idx,
                    section_idx=sec_idx,
                )
            )
    return chunks
