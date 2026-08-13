"""Raw LS-DYNA card preservation — TRD_LEVEL3.md §7, PRD_LEVEL3.md §11.

A `RawCard` is one `*KEYWORD` occurrence plus every line that follows it up
to (not including) the next `*`-starting line or end of file. This is the
unit every entity in `packages/cae/lsdyna/models.py` is derived from, and it
is *never discarded* — even a keyword this parser has no structured
understanding of still becomes a `RawCard` with `raw_text` intact
(Instructions §8: "Unknown keyword: preserve raw content, never invent
engineering meaning").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class RawCard:
    keyword: str  # normalized: uppercase, no leading "*", e.g. "MAT_024"
    raw_text: str  # the keyword line + all its data/comment lines, verbatim
    source_file: str
    line_start: int  # 1-indexed, inclusive
    line_end: int  # 1-indexed, inclusive
    raw_hash: str  # sha256(raw_text)

    @staticmethod
    def build(keyword: str, raw_text: str, source_file: str, line_start: int, line_end: int) -> RawCard:
        return RawCard(
            keyword=keyword,
            raw_text=raw_text,
            source_file=source_file,
            line_start=line_start,
            line_end=line_end,
            raw_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        )

    def data_lines(self) -> list[str]:
        """Every line after the keyword line, with full-line `$` comments
        dropped — inline data is never dropped, only lines that are
        entirely a comment."""
        lines = self.raw_text.splitlines()[1:]
        return [line for line in lines if not line.strip().startswith("$")]
