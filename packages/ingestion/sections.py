"""Heading detection -> section boundaries.

A deliberately simple heuristic (numbered headings like "3.2.1 Contact
Definitions", or short ALL-CAPS lines) — not a layout model. Good enough to
give chunks a section/page anchor for citation (PRD.md PR-011); a
mis-detected heading just means a slightly larger or smaller section, never
fabricated content, so the failure mode is benign.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from packages.ingestion.extract import PageExtraction

_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})\s+([A-Z][A-Za-z0-9 ,\-/()]{3,80})$")
_CAPS_HEADING = re.compile(r"^([A-Z][A-Z0-9 \-/]{6,80})$")


@dataclass
class DetectedSection:
    title: str
    section_number: str | None
    start_page: int
    end_page: int
    paragraphs: list[tuple[int, str]] = field(default_factory=list)  # (page_number, paragraph_text)


def _match_heading(line: str) -> tuple[str, str | None] | None:
    line = line.strip()
    if not line or len(line) > 90:
        return None
    m = _NUMBERED_HEADING.match(line)
    if m:
        return m.group(2).strip(), m.group(1)
    m = _CAPS_HEADING.match(line)
    if m:
        return m.group(1).strip(), None
    return None


def detect_sections(pages: list[PageExtraction]) -> list[DetectedSection]:
    if not pages:
        return []

    sections: list[DetectedSection] = []
    current = DetectedSection(
        title="Front matter", section_number=None, start_page=pages[0].page_number, end_page=pages[0].page_number
    )

    for page in pages:
        for raw_line in page.text.splitlines():
            heading = _match_heading(raw_line)
            if heading is not None:
                current.end_page = page.page_number
                sections.append(current)
                title, number = heading
                current = DetectedSection(
                    title=title, section_number=number, start_page=page.page_number, end_page=page.page_number
                )
                continue
            text = raw_line.strip()
            if text:
                current.paragraphs.append((page.page_number, text))
        current.end_page = page.page_number

    sections.append(current)
    # Drop an empty leading placeholder (no heading matched before real content).
    return [s for s in sections if s.paragraphs or s.section_number]
