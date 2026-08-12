"""Integration tests against the real ingested/indexed corpus
(scripts/ingest_documents.py + scripts/index_knowledge.py must have run —
see tests/conftest.py's shared-DB caveat).
"""

from sqlalchemy.orm import Session

from packages.retrieval.index import index_chunks
from packages.retrieval.search import SourceFilter, reciprocal_rank_fusion, retrieve
from tests.conftest import requires_db


def test_reciprocal_rank_fusion_pure() -> None:
    import uuid

    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    scores = reciprocal_rank_fusion([[a, b, c], [b, a]])
    # b appears at rank 1 in the second list and rank 2 in the first —
    # should outrank a (rank 1 + rank 2 vs rank 2 + rank 1... actually tied)
    assert set(scores) == {a, b, c}
    assert scores[a] > scores[c]
    assert scores[b] > scores[c]


@requires_db
def test_retrieve_finds_un_r94_for_frontal_collision(session: Session) -> None:
    index_chunks(session)  # idempotent no-op if scripts/index_knowledge.py already ran

    results = retrieve(session, "frontal collision occupant protection", limit=5)
    assert results, "expected at least one retrieved chunk"
    assert any(r.document_key == "unece-un-r94" for r in results)

    top = results[0]
    assert top.authority_level in ("AUTHORITATIVE", "OFFICIAL_DOCUMENTATION")
    assert top.page_start is not None


@requires_db
def test_retrieve_respects_source_type_filter(session: Session) -> None:
    index_chunks(session)

    results = retrieve(session, "contact definition", filters=SourceFilter(source_type="REGULATION"), limit=10)
    assert all(r.source_type == "REGULATION" for r in results)


@requires_db
def test_retrieve_marks_which_leg_matched(session: Session) -> None:
    index_chunks(session)

    results = retrieve(session, "dummy chest deflection", limit=5)
    assert results
    assert all(r.matched_fts or r.matched_vector for r in results)
