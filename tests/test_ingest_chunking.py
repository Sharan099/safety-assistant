"""Unit tests for clause and table chunking helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.clause_splitter import is_page_section_title, split_body_by_clauses  # noqa: E402
from ingestion.table_chunker import extract_tables_from_body  # noqa: E402


def test_page_section_detection():
    assert is_page_section_title("Page 42")
    assert not is_page_section_title("§5.2.1.4")


def test_split_unece_clause_lines():
    body = "5.2.1.4.\nThe Thorax Compression Criterion (ThCC) shall not exceed 42 mm;\n5.2.1.5.\nNext clause."
    blocks = split_body_by_clauses(body, page_number=10)
    assert len(blocks) >= 2
    assert blocks[0][0] == "5.2.1.4"
    assert "42 mm" in blocks[0][2]
    assert blocks[0][3] == 10


def test_extract_pipe_table_atomic():
    body = "Intro text\n| A | B |\n| --- | --- |\n| 1 | 2 |\nAfter table"
    remainder, tables = extract_tables_from_body(body, clause_number="6.4")
    assert len(tables) == 1
    assert "| A | B |" in tables[0].markdown
    assert "Intro text" in tables[0].preamble
    assert "After table" in remainder
