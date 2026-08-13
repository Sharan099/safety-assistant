"""LS-DYNA lexer — splits raw deck text into `RawCard`s.

A line starting with `*` opens a new card that runs until the next
`*`-starting line or end of file. Lines before the first `*` line (rare —
sometimes a leading `$` banner comment) are not part of any card and are
dropped; nothing structural is lost since they carry no keyword to attach
to and downstream fields never sees them.

Deliberately dumb: this module has no idea what any keyword *means*. That
boundary is what lets `packages/cae/lsdyna/parser.py` add real per-keyword
extraction later without ever having to touch tokenization again.
"""

from __future__ import annotations

from packages.cae.lsdyna.cards import RawCard


def _normalize_keyword(line: str) -> str:
    return line.strip().lstrip("*").strip().upper()


def tokenize(text: str, source_file: str) -> list[RawCard]:
    lines = text.splitlines()
    cards: list[RawCard] = []

    current_keyword: str | None = None
    current_start = 0
    current_lines: list[str] = []

    def _flush(end_line: int) -> None:
        if current_keyword is None:
            return
        raw_text = "\n".join(current_lines)
        cards.append(RawCard.build(current_keyword, raw_text, source_file, current_start, end_line))

    for i, line in enumerate(lines, start=1):
        if line.strip().startswith("*"):
            _flush(i - 1)
            current_keyword = _normalize_keyword(line)
            current_start = i
            current_lines = [line]
        else:
            if current_keyword is not None:
                current_lines.append(line)
            # else: pre-keyword banner content, intentionally dropped (see
            # module docstring).

    _flush(len(lines))
    return cards
