"""Real BM25 — TRD_LEVEL3.md §16/§21/§33, docs/ADR/0012."""

import uuid

from sqlalchemy.orm import Session

from packages.retrieval.bm25 import Bm25Index, bm25_search, build_bm25_index, tokenize
from packages.retrieval.index import index_chunks
from tests.legacy.conftest import requires_db


def test_tokenize_preserves_asterisk_and_underscore_for_cae_identifiers() -> None:
    tokens = tokenize("What does *DATABASE_BINARY_D3PLOT control?")
    assert "*database_binary_d3plot" in tokens


def _build_index(docs: list[str]) -> Bm25Index:
    from rank_bm25 import BM25Okapi

    chunk_ids = [uuid.uuid4() for _ in docs]
    corpus = [tokenize(d) for d in docs]
    return Bm25Index(chunk_ids=chunk_ids, bm25=BM25Okapi(corpus))


def test_bm25_search_ranks_exact_term_match_first() -> None:
    docs = [
        "The *CONTACT_AUTOMATIC_SURFACE_TO_SURFACE keyword defines a contact interface.",
        "Frontal collision protection requirements for occupant restraint systems.",
        "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE is the most common contact type in LS-DYNA.",
    ]
    index = _build_index(docs)
    ranked = bm25_search(index, "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE", limit=10)
    assert len(ranked) == 2  # only the two docs that actually contain the term
    assert set(ranked) == {index.chunk_ids[0], index.chunk_ids[2]}


def test_bm25_search_excludes_zero_score_documents() -> None:
    docs = ["Completely unrelated document about seat belts.", "Another unrelated document about airbags."]
    index = _build_index(docs)
    ranked = bm25_search(index, "*MAT_024 material properties", limit=10)
    assert ranked == []


def test_bm25_search_empty_index_returns_empty() -> None:
    empty = Bm25Index(chunk_ids=[], bm25=None)
    assert bm25_search(empty, "anything", limit=10) == []


def test_bm25_search_empty_query_returns_empty() -> None:
    index = _build_index(["some content here"])
    assert bm25_search(index, "", limit=10) == []


def test_bm25_search_respects_limit() -> None:
    docs = [f"contact definition example number {i}" for i in range(10)]
    index = _build_index(docs)
    ranked = bm25_search(index, "contact definition", limit=3)
    assert len(ranked) == 3


@requires_db
def test_build_bm25_index_against_real_corpus_finds_frontal_collision(session: Session) -> None:
    index_chunks(session)  # idempotent no-op if scripts/index_knowledge.py already ran
    index = build_bm25_index(session)
    assert index.bm25 is not None
    assert len(index.chunk_ids) > 0
    ranked = bm25_search(index, "frontal collision occupant protection", limit=5)
    assert ranked
