"""Phase 2 — chunk quality and UN R94 chest deflection retrieval acceptance."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CHUNKS_FILE  # noqa: E402

CHEST_RE = re.compile(r"chest\s+deflection|thorax\s+compression|thcc", re.I)
MM_RE = re.compile(r"\d+(?:\.\d+)?\s*mm", re.I)


@pytest.fixture(scope="module")
def chunks():
    data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return data["chunks"]


@pytest.fixture(scope="module")
def retriever():
    from backend.app.retrieval.hybrid import HybridRetriever
    return HybridRetriever()


def test_chunks_have_metadata_schema(chunks):
    sample = chunks[0]
    for field in ("doc_id", "doc_type", "authority", "region", "test_type"):
        assert field in sample


def test_corpus_has_un_r94_thorax_limit(chunks):
    """Corpus must contain a UN R94 thorax/chest limit with mm and clause."""
    r94 = [c for c in chunks if c.get("regulation") == "UN_R94"]
    found = any(
        CHEST_RE.search(c.get("text", ""))
        and MM_RE.search(c.get("text", ""))
        and (c.get("clause") or c.get("clause_number"))
        for c in r94
    )
    assert found, "UN R94 corpus missing thorax/chest limit chunk with mm + clause"


def test_un_r94_chest_deflection_retrieval(retriever, chunks):
    """Retrieval returns UN R94 injury-limit content for chest deflection query."""
    test_corpus_has_un_r94_thorax_limit(chunks)
    result = retriever.retrieve(
        "What is the chest deflection limit under UN R94?"
    )
    assert result["documents"], "No documents retrieved"
    found = False
    for d in result["documents"][:12]:
        chunk = retriever._chunk_by_id.get(d["id"], {})
        text = chunk.get("text", "") + " " + (d.get("parent_context") or "")
        is_r94 = chunk.get("regulation") == "UN_R94"
        has_injury = bool(CHEST_RE.search(text)) or bool(MM_RE.search(text))
        has_clause = bool(chunk.get("clause") or chunk.get("clause_number"))
        if is_r94 and has_injury and (has_clause or MM_RE.search(text)):
            found = True
            break
    assert found, "No UN R94 injury-limit chunk in top 12"


def test_quality_gate_rejects_tiny_file(tmp_path):
    from ingestion.quality_gate import check_markdown

    p = tmp_path / "tiny.md"
    p.write_text("x" * 50, encoding="utf-8")
    qr = check_markdown(p)
    assert not qr.passed
