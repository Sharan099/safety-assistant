"""Part B: atomic table chunks with pipe markdown."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.table_structure_enrichment import enrich_table_chunks


def test_annex6_emits_table_chunk_type():
    sparse = [{
        "regulation": "UN_R14",
        "clause": "Annex 6",
        "section_title": "Annex 6",
        "text": "# UN_R14 > Annex 6\n\nM1 front outboard 3 anchorages",
        "word_count": 10,
    }]
    out = enrich_table_chunks(sparse)
    table = next(c for c in out if c.get("chunk_type") == "table")
    assert table.get("table_structured")
    assert "| M1 |" in table["text"]
    assert "Ø" in table["text"] or "╬" in table["text"]
