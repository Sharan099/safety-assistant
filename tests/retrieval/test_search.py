"""Integration tests against the real ingested/indexed corpus
(scripts/ingest_documents.py + scripts/index_knowledge.py must have run —
see tests/conftest.py's shared-DB caveat).
"""

import uuid

from sqlalchemy.orm import Session

from packages.domain.knowledge import DocumentChunk, DocumentRevision, Embedding
from packages.retrieval.embeddings import HashingEmbeddingProvider
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


@requires_db
def test_retrieve_finds_lsdyna_for_airbag_query(session: Session) -> None:
    index_chunks(session)

    results = retrieve(session, "airbag liner reference geometry venting leakage of gas", limit=5)
    assert results
    assert any(r.document_key == "lsdyna-r17-theory" for r in results)


@requires_db
def test_retrieve_rejects_deliberately_irrelevant_chunk(session: Session) -> None:
    """CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 8: insert a chunk with
    content sharing no real topic with an engineering query (same category
    of false positive the PRD addendum reported: an AES/Rijndael passage)
    into a real document revision, embed it, and confirm retrieve() excludes
    it — while still returning the genuinely relevant corpus content.
    """
    index_chunks(session)

    revision = session.query(DocumentRevision).filter(DocumentRevision.status == "READY").first()
    assert revision is not None, "expected at least one ingested revision (run scripts/ingest_documents.py)"

    irrelevant_chunk = DocumentChunk(
        document_revision_id=revision.id,
        chunk_type="text",
        content=(
            "The Advanced Encryption Standard (AES), also known as Rijndael, is a "
            "specification for the encryption of electronic data using a "
            "substitution-permutation network with 128, 192, or 256 bit keys."
        ),
        token_count=30,
        source_locator={"page_start": 1, "page_end": 1, "section": "__test_irrelevant"},
    )
    session.add(irrelevant_chunk)
    session.flush()

    provider = HashingEmbeddingProvider()
    session.add(
        Embedding(
            chunk_id=irrelevant_chunk.id,
            model_name=provider.model_name,
            model_version=provider.model_version,
            dimensions=provider.dimensions,
            embedding=provider.embed(irrelevant_chunk.content),
        )
    )
    session.flush()

    results = retrieve(session, "frontal collision occupant protection restraint belt force limiter", limit=10)
    assert results, "guard should not reject everything — relevant corpus content must still come back"
    assert irrelevant_chunk.id not in {r.chunk_id for r in results}


@requires_db
def test_retrieve_still_finds_uuid_typed_results(session: Session) -> None:
    # Sanity: RetrievedChunk.chunk_id really is a uuid.UUID, not a string —
    # the dedup/guard test above compares it directly against DB-side ids.
    index_chunks(session)
    results = retrieve(session, "frontal collision", limit=1)
    assert results
    assert isinstance(results[0].chunk_id, uuid.UUID)
