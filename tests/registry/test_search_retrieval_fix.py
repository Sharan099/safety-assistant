"""Retrieval fix regression tests (Phase 2.5)."""

from __future__ import annotations

from unittest.mock import patch

from registry.search import RegulationSearchEngine


def test_single_unece_reg_does_not_set_redundant_source_type():
    engine = RegulationSearchEngine()
    filters = engine._parse_query_for_filters(
        "What is the chest deflection limit under UN R94?"
    )
    assert filters["regulation_code"] == "UN_R94"
    assert "source_type" not in filters


def test_r16_762_crossref_sets_r16_filter_only():
    engine = RegulationSearchEngine()
    filters = engine._parse_query_for_filters(
        "the locking mechanism is deceleration-actuated only — check this against §7.6.2"
    )
    assert filters["regulation_code"] == "UN_R16"
    assert "source_type" not in filters


def test_multi_reg_query_sets_unece_filter():
    engine = RegulationSearchEngine()
    filters = engine._parse_query_for_filters(
        "What is the difference between UN R44 and UN R129 for child restraints?"
    )
    assert filters["source_type"] == "UNECE"
    assert set(filters["regulation_codes"]) == {"UN_R44", "UN_R129"}


def test_r94_chest_query_expansion():
    engine = RegulationSearchEngine()
    expanded = engine._expand_retrieval_query(
        "What is the chest deflection limit under UN R94?",
        {"regulation_code": "UN_R94"},
    )
    assert "ThCC" in expanded
    assert "5.2.1.4" in expanded


def test_mock_reranker_uses_rrf_order(db_session):
    engine = RegulationSearchEngine()
    low = {"chunk_id": 1, "chunk_text": "low rrf", "rrf_score": 0.1, "search_score": 0.1}
    high = {"chunk_id": 2, "chunk_text": "high rrf ThCC 42 mm", "rrf_score": 0.9, "search_score": 0.9}
    fused = [high, low]
    engine.reranker.use_mock_reranker = True
    with patch.object(engine, "_dense_search_sqlite", return_value=[low]):
        with patch.object(engine, "_sparse_search_sqlite", return_value=[high]):
            with patch.object(engine, "_reciprocal_rank_fusion", return_value=fused):
                with patch.object(engine.embedder, "embed_query", return_value=[0.0] * 768):
                    with patch.object(
                        engine,
                        "_generate_grounded_answer",
                        return_value=("ans", {"model_key": "test"}),
                    ):
                        out = engine.search(
                            db_session,
                            "What is the chest deflection limit under UN R94?",
                            top_k=2,
                            rerank=True,
                        )
    assert out["sources"][0]["chunk_id"] == 2


def test_r14_isofix_query_expansion():
    engine = RegulationSearchEngine()
    expanded = engine._expand_retrieval_query(
        "What does UN R14 Annex 6 specify for ISOFIX anchorage points?",
        {"regulation_code": "UN_R14"},
    )
    assert "Annex 6" in expanded
    assert "Vehicle category" in expanded


def test_promote_named_annex_injects_table_chunk():
    engine = RegulationSearchEngine()
    table = {
        "chunk_id": 15848,
        "chunk_text": "Vehicle category M1 M2 anchorage table",
        "section": "Annex_6",
        "chunk_type": "table",
        "rrf_score": 0.015,
    }
    cross_ref = {
        "chunk_id": 1,
        "chunk_text": "specified in Annex 6",
        "section": "5.3.2",
        "chunk_type": "clause",
        "rrf_score": 0.03,
    }
    filler = [
        {"chunk_id": i, "chunk_text": f"f{i}", "section": "5", "rrf_score": 0.02}
        for i in range(2, 10)
    ]
    fused = [cross_ref, *filler, table]
    top = engine._promote_named_annex_chunks(
        "What does UN R14 Annex 6 specify for ISOFIX anchorage points?",
        fused,
        top_k=8,
    )
    assert any(c["chunk_id"] == 15848 for c in top)
    assert top[0]["chunk_id"] == 1  # higher RRF retained at front


def test_parent_expansion_adds_parent(db_session):
    from database.models import Chunk, Document, Regulation

    reg = Regulation(
        id=1,
        regulation_code="UN_R94",
        title="R94",
        source_type="UNECE",
        status="ACTIVE",
    )
    doc = Document(
        id=1,
        regulation_id=1,
        document_name="UN_R94.pdf",
        document_type="PDF",
        file_path="/x.pdf",
        hash="abc",
    )
    parent = Chunk(
        id=10,
        document_id=1,
        chunk_text="parent section with ThCC 42 mm and 5.2.1.4",
        chunk_type="section",
        chunk_index=0,
    )
    child = Chunk(
        id=11,
        document_id=1,
        chunk_text="child clause",
        chunk_type="clause",
        parent_chunk_id=10,
        chunk_index=1,
    )
    db_session.add_all([reg, doc, parent, child])
    db_session.commit()

    engine = RegulationSearchEngine()
    expanded = engine._expand_with_parents(
        db_session,
        [engine._format_chunk_item(child, 0.9)],
        max_extra=2,
    )
    ids = [c["chunk_id"] for c in expanded]
    assert 11 in ids and 10 in ids


def test_cross_reference_expansion(db_session):
    from database.models import Chunk, Document, Regulation
    import os

    reg = Regulation(
        id=2,
        regulation_code="UN_R16_TEST",
        title="R16 Test",
        source_type="UNECE",
        status="ACTIVE",
    )
    doc = Document(
        id=2,
        regulation_id=2,
        document_name="UN_R16_TEST.pdf",
        document_type="PDF",
        file_path="/y.pdf",
        hash="def",
    )
    c1 = Chunk(
        id=20,
        document_id=2,
        chunk_text="Refer to paragraph 2.12.4.1 for details.",
        chunk_type="clause",
        section="5.8",
        chunk_index=0,
    )
    c2 = Chunk(
        id=21,
        document_id=2,
        chunk_text="Emergency locking retractor specifications.",
        chunk_type="clause",
        section="2.12.4",
        chunk_index=1,
    )
    db_session.add_all([reg, doc, c1, c2])
    db_session.commit()

    engine = RegulationSearchEngine()
    os.environ["ENABLE_CROSS_REFERENCE_EXPANSION"] = "true"
    try:
        expanded = engine._expand_cross_references(
            db_session,
            [engine._format_chunk_item(c1, 0.9)],
            max_depth=2,
            max_expansion=2
        )
        ids = [c["chunk_id"] for c in expanded]
        assert 20 in ids
        assert 21 in ids  # resolved transitively via paragraph 2.12.4.1 -> section 2.12.4
    finally:
        os.environ["ENABLE_CROSS_REFERENCE_EXPANSION"] = "false"


def test_deprioritize_scope_chunks():
    engine = RegulationSearchEngine()
    scope = {"chunk_id": 1, "section": "1", "chunk_text": "As defined in the Consolidated Resolution"}
    substance = {"chunk_id": 2, "section": "5.2.1.4", "chunk_text": "ThCC shall not exceed 42 mm"}
    reordered = engine._deprioritize_scope_chunks([scope, substance])
    assert reordered[0]["chunk_id"] == 2


def test_promote_cited_section_injects_target():
    engine = RegulationSearchEngine()
    filler = [
        {"chunk_id": i, "section": "6.2", "chunk_text": f"f{i}", "rrf_score": 0.02}
        for i in range(2, 10)
    ]
    target = {
        "chunk_id": 99,
        "section": "7.6.2",
        "chunk_text": "locking test emergency locking retractor",
        "rrf_score": 0.01,
    }
    fused = filler + [target]
    top = engine._promote_cited_section_chunks(
        "check against §7.6.2 locking retractor", fused, top_k=8
    )
    assert any(c["chunk_id"] == 99 for c in top)


def test_r16_762_query_expansion():
    engine = RegulationSearchEngine()
    expanded = engine._expand_retrieval_query(
        "the locking mechanism is deceleration-actuated only — check this against §7.6.2",
        {"regulation_code": "UN_R16"},
    )
    assert "7.6.2" in expanded
    assert "emergency locking retractor" in expanded

