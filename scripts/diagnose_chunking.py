#!/usr/bin/env python3
"""
Read-only diagnostic report for output/regulation_chunks.json.

Usage:
  python scripts/diagnose_chunking.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CHUNKS_FILE, CHUNKING_DIAGNOSTICS  # noqa: E402

REPORT_FILE = CHUNKING_DIAGNOSTICS

CLAUSE_MARKER_RE = re.compile(r"^\(?[a-zA-Z0-9]+[\.\)]\s", re.MULTILINE)
STRUCTURE_HEADING_RE = re.compile(
    r"^(Article|Section|Annex|Part)\s+\w+",
    re.MULTILINE | re.IGNORECASE,
)


class Report:
    def __init__(self) -> None:
        self._buf = StringIO()

    def line(self, text: str = "") -> None:
        print(text)
        self._buf.write(text + "\n")

    def header(self, title: str, char: str = "=") -> None:
        self.line()
        self.line(title)
        self.line(char * len(title))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._buf.getvalue(), encoding="utf-8")


def _pct(part: int, whole: int) -> float:
    return (100.0 * part / whole) if whole else 0.0


def _truncate(text: str, n: int = 200) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def clause_coverage_by_regulation(chunks: list[dict], r: Report) -> list[tuple]:
    r.header("1. Clause number coverage by regulation")

    by_reg: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_reg[c.get("regulation") or "UNKNOWN"].append(c)

    rows: list[tuple[str, int, int, int, float, float]] = []
    for reg, reg_chunks in by_reg.items():
        total = len(reg_chunks)
        with_clause = sum(1 for c in reg_chunks if c.get("clause_number"))
        without = total - with_clause
        rows.append(
            (
                reg,
                total,
                with_clause,
                without,
                _pct(with_clause, total),
                _pct(without, total),
            )
        )

    rows.sort(key=lambda x: x[5], reverse=True)

    r.line(
        f"{'Regulation':<22} {'Total':>8} {'w/ clause':>10} {'%':>7} "
        f"{'null':>8} {'% null':>8}"
    )
    r.line("-" * 68)
    for reg, total, with_c, without, pct_with, pct_null in rows:
        r.line(
            f"{reg:<22} {total:>8} {with_c:>10} {pct_with:>6.1f}% "
            f"{without:>8} {pct_null:>7.1f}%"
        )

    return rows


def sample_null_clause_evidence(
    chunks: list[dict], worst_reg: str, r: Report
) -> None:
    r.header(f"2. Sample null-clause chunks - worst offender: {worst_reg}")

    reg_null = [
        c for c in chunks if c.get("regulation") == worst_reg and not c.get("clause_number")
    ]
    if not reg_null:
        r.line("No null-clause chunks for this regulation.")
        return

    n = len(reg_null)
    if n <= 15:
        indices = list(range(n))
    else:
        starts = [0, 1, 2, 3, 4]
        mid = n // 2
        middles = [mid - 2, mid - 1, mid, mid + 1, mid + 2]
        ends = [n - 5, n - 4, n - 3, n - 2, n - 1]
        indices = sorted(set(starts + middles + ends))[:15]

    r.line(f"Picked {len(indices)} examples from {n} null-clause chunks in {worst_reg}")
    r.line()
    for i, idx in enumerate(indices, 1):
        c = reg_null[idx]
        r.line(f"--- Example {i} (index {idx} in null-clause list) ---")
        r.line(f"  chunk_id:     {c.get('chunk_id')}")
        r.line(f"  heading_path: {c.get('heading_path')}")
        r.line(f"  text (200c):  {_truncate(c.get('text', ''), 200)}")
        r.line()


def truncated_parent_analysis(chunks: list[dict], r: Report) -> None:
    r.header("3. Truncated parent section analysis")

    sections = [c for c in chunks if c.get("chunk_type") == "section"]
    total_sec = len(sections)
    truncated = [c for c in sections if c.get("is_truncated_parent") is True]
    n_trunc = len(truncated)
    r.line(f"Section chunks (denominator):     {total_sec}")
    r.line(f"is_truncated_parent == true:      {n_trunc} ({_pct(n_trunc, total_sec):.1f}%)")
    r.line()

    by_reg: dict[str, int] = defaultdict(int)
    for c in truncated:
        by_reg[c.get("regulation") or "UNKNOWN"] += 1

    r.line("Truncated section counts by regulation (descending):")
    r.line(f"  {'Regulation':<22} {'Count':>8}")
    r.line("  " + "-" * 32)
    for reg, count in sorted(by_reg.items(), key=lambda x: x[1], reverse=True):
        r.line(f"  {reg:<22} {count:>8}")
    r.line()

    top5 = sorted(truncated, key=lambda c: c.get("word_count") or 0, reverse=True)[:5]
    r.line("Top 5 truncated sections by word_count:")
    for i, c in enumerate(top5, 1):
        r.line(f"  {i}. chunk_id:     {c.get('chunk_id')}")
        r.line(f"     heading_path: {c.get('heading_path')}")
        r.line(f"     word_count:   {c.get('word_count')}")
        r.line()


def heading_style_sniff(chunks: list[dict], r: Report) -> None:
    r.header("4. Heading style sniff test (null clause_number chunks only)")

    null_chunks = [c for c in chunks if not c.get("clause_number")]
    r.line(f"Scanning {len(null_chunks)} chunks with null clause_number")
    r.line()

    examples: list[tuple[str, str, str]] = []  # chunk_id, pattern, line

    for c in null_chunks:
        text = c.get("text") or ""
        cid = c.get("chunk_id", "")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if CLAUSE_MARKER_RE.match(line):
                examples.append((cid, "clause_marker", line))
            elif STRUCTURE_HEADING_RE.match(line):
                examples.append((cid, "structure_heading", line))
            if len(examples) >= 20:
                break
        if len(examples) >= 20:
            break

    if not examples:
        r.line("No suspicious heading-like lines found in null-clause chunks.")
        return

    r.line(f"Up to {len(examples)} example lines that may be missed structure:")
    r.line(f"  {'Pattern':<18} {'chunk_id':<42} Line")
    r.line("  " + "-" * 100)
    for cid, pattern, line in examples[:20]:
        display = _truncate(line, 80)
        r.line(f"  {pattern:<18} {cid:<42} {display}")


def main() -> int:
    if not CHUNKS_FILE.exists():
        print(f"ERROR: {CHUNKS_FILE} not found", file=sys.stderr)
        return 1

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    chunks: list[dict] = data.get("chunks", [])
    if not chunks:
        print("ERROR: no chunks in file", file=sys.stderr)
        return 1

    r = Report()
    r.header("Chunking diagnostics report", char="=")
    r.line(f"Source: {CHUNKS_FILE}")
    r.line(f"Total chunks loaded: {len(chunks)}")

    coverage_rows = clause_coverage_by_regulation(chunks, r)

    worst_reg = coverage_rows[0][0] if coverage_rows else "UNKNOWN"
    sample_null_clause_evidence(chunks, worst_reg, r)

    truncated_parent_analysis(chunks, r)
    heading_style_sniff(chunks, r)

    r.save(REPORT_FILE)
    r.line()
    r.line(f"Report also written to: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
