"""Lightweight LS-DYNA keyword scanning — PRD_LEVEL3.md §13,
CLAUDE_CODE_LEVEL3_INSTRUCTIONS.md §5/§7: "Do not perform full semantic
parsing in the profiler."

This module counts keyword occurrences by regex line-scan only. It never
interprets card fields, never resolves entity relationships, and never
follows `*INCLUDE`. That is `packages/cae/lsdyna/` (the real parser)'s job.
A "PART count" here means "number of `*PART` keyword lines", which is a
close but not exact proxy for the number of parts actually defined (a
`*PART` block can define more than one part in rare cases) — documented
here rather than silently treated as exact.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

from packages.cae.lsdyna.registry import registry_roots
from packages.cae.lsdyna.registry import root_for as _registry_root_for

_KEYWORD_LINE_RE = re.compile(r"^\*([A-Za-z0-9_]+)")

# Filename/path tokens -> free-text model hint. Heuristic only, per
# PRD_LEVEL3.md §13 "Model hints" column — never treated as a validated
# classification.
_MODEL_HINT_TOKENS = {
    "SEAT": "Seat",
    "BIORID": "BioRID",
    "THOR": "THOR",
    "DUMMY": "Dummy",
    "RESTRAINT": "Restraint",
    "BELT": "Belt/restraint",
    "AIRBAG": "Airbag",
    "SLED": "Sled buck",
    "VEHICLE": "Vehicle",
    "SILVERADO": "Vehicle (Silverado)",
    "ACCORD": "Vehicle (Accord)",
    "NEON": "Vehicle (Neon)",
    "YARIS": "Vehicle (Yaris)",
}


@dataclass
class KeywordScanResult:
    keyword_counts: dict[str, int] = field(default_factory=dict)
    include_count: int = 0
    root_counts: dict[str, int] = field(default_factory=dict)  # per keyword_registry.yaml's roots
    model_hints: list[str] = field(default_factory=list)
    line_count: int = 0


def scan_keywords(text: str) -> KeywordScanResult:
    result = KeywordScanResult()
    roots = registry_roots()
    for line in text.splitlines():
        result.line_count += 1
        stripped = line.strip()
        if not stripped.startswith("*") or stripped.startswith("$"):
            continue
        match = _KEYWORD_LINE_RE.match(stripped)
        if not match:
            continue
        keyword = match.group(1).upper()
        result.keyword_counts[keyword] = result.keyword_counts.get(keyword, 0) + 1

        root = _registry_root_for(keyword, roots)
        if root is not None:
            result.root_counts[root] = result.root_counts.get(root, 0) + 1
        if root == "INCLUDE":
            result.include_count += 1
    return result


def model_hints_for_path(path_str: str) -> list[str]:
    hint_source = path_str.upper()
    return sorted({hint for token, hint in _MODEL_HINT_TOKENS.items() if token in hint_source})


def scan_file(path: pathlib.Path) -> KeywordScanResult:
    """Reads with errors="replace" — LS-DYNA decks are not reliably UTF-8
    (fixed-width legacy exports can contain stray bytes); a scan must never
    crash on that, only ever undercount a malformed line."""
    text = path.read_text(encoding="utf-8", errors="replace")
    result = scan_keywords(text)
    result.model_hints = model_hints_for_path(path.as_posix())
    return result
