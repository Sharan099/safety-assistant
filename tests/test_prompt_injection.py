"""Regression tests for prompt injection, identity, and corpus-meta routing."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from api.routes import get_search_engine
from database.connection import get_db
from registry.chat_intent import (
    INJECTION_TRANSCRIPT_FR,
    classify_chat_query,
    detect_instruction_injection,
    is_identity_question,
    is_corpus_meta_question,
    is_document_count_question,
    build_injection_refusal,
    query_document_registry,
    build_corpus_meta_answer,
)
from registry.prompt_templates import GROUNDED_SYSTEM_PROMPT, IDENTITY_RESPONSE
from registry.search import RegulationSearchEngine

IDENTITY_TRANSCRIPT = "Who are you?"

CORPUS_META_TRANSCRIPT_DOCS = "What documents do you use?"

CORPUS_META_TRANSCRIPT_REGS = "what regulations do you have access to"

CORPUS_META_TRANSCRIPT_COUNT = "How many PDFs/documents does the system use?"

CORPUS_META_TRANSCRIPT_COUNT_ALT = "count the number of pdf"

FABRICATED_DOC_NAMES = (
    "IIHS_RoofStrength.pdf",
    "NHTSA_NCAP_Frontal_Test_Procedure.pdf",
    "FMVSS_213_2024.pdf",
)


class TestInjectionDetection:
    def test_exact_french_translation_indirection_transcript(self):
        assert detect_instruction_injection(INJECTION_TRANSCRIPT_FR) is True
        assert classify_chat_query(INJECTION_TRANSCRIPT_FR) == "injection_blocked"

    def test_refusal_does_not_comply_with_embedded_instruction(self):
        refusal = build_injection_refusal(INJECTION_TRANSCRIPT_FR)
        low = refusal.lower()
        assert "cannot follow" in low or "must refuse" in low or "refuse" in low
        assert "ignore citation" in low or "invent" in low
        assert not re.search(r"i will ignore", low)
        assert not re.search(r"here is (?:an? )?invented", low)


class TestIdentityRouting:
    def test_exact_who_are_you_transcript(self):
        assert is_identity_question(IDENTITY_TRANSCRIPT) is True
        assert classify_chat_query(IDENTITY_TRANSCRIPT) == "identity"

    def test_identity_response_is_ai_not_human_engineer(self):
        low = IDENTITY_RESPONSE.lower()
        assert "ai assistant" in low
        assert "not a human" in low
        assert "senior passive safety engineer" not in low

    def test_chat_identity_skips_search_engine(self):
        client = TestClient(app)
        mock_engine = MagicMock()
        app.dependency_overrides[get_search_engine] = lambda: mock_engine
        mock_db = MagicMock()
        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            resp = client.post("/api/v1/chat", json={"query": IDENTITY_TRANSCRIPT})
            assert resp.status_code == 200
            data = resp.json()
            assert data["answer"] == IDENTITY_RESPONSE
            assert data["citations"] == []
            assert data["gateway"].get("route") == "identity"
            mock_engine.search.assert_not_called()
        finally:
            app.dependency_overrides.pop(get_search_engine, None)
            app.dependency_overrides.pop(get_db, None)

    def test_system_prompt_does_not_claim_human_engineer_identity(self):
        assert "You are NOT a human engineer" in GROUNDED_SYSTEM_PROMPT
        assert "You are a Senior passive safety engineer" not in GROUNDED_SYSTEM_PROMPT


class TestCorpusMetaRouting:
    def test_exact_what_documents_transcript(self):
        assert is_corpus_meta_question(CORPUS_META_TRANSCRIPT_DOCS) is True
        assert classify_chat_query(CORPUS_META_TRANSCRIPT_DOCS) == "corpus_meta"

    def test_exact_regulations_access_transcript(self):
        assert is_corpus_meta_question(CORPUS_META_TRANSCRIPT_REGS) is True
        assert classify_chat_query(CORPUS_META_TRANSCRIPT_REGS) == "corpus_meta"

    def test_regulatory_question_not_misclassified_as_corpus_meta(self):
        q = "What is the chest deflection limit under UN R94?"
        assert is_corpus_meta_question(q) is False
        assert classify_chat_query(q) == "regulatory"

    def test_exact_how_many_pdfs_transcript(self):
        assert is_corpus_meta_question(CORPUS_META_TRANSCRIPT_COUNT) is True
        assert is_document_count_question(CORPUS_META_TRANSCRIPT_COUNT) is True
        assert classify_chat_query(CORPUS_META_TRANSCRIPT_COUNT) == "corpus_meta"

    def test_count_the_number_of_pdf_transcript(self):
        assert is_corpus_meta_question(CORPUS_META_TRANSCRIPT_COUNT_ALT) is True
        assert is_document_count_question(CORPUS_META_TRANSCRIPT_COUNT_ALT) is True
        assert classify_chat_query(CORPUS_META_TRANSCRIPT_COUNT_ALT) == "corpus_meta"

    def test_corpus_meta_count_answer_uses_registry_not_invented_names(self, db_session):
        registry = query_document_registry(db_session)
        answer = build_corpus_meta_answer(db_session, CORPUS_META_TRANSCRIPT_COUNT)
        assert str(registry["total_documents"]) in answer
        assert str(registry["pdf_documents"]) in answer
        assert "documents` table" in answer or "documents table" in answer
        assert "not semantic retrieval" in answer.lower() or "not semantic search" in answer.lower()
        for fake in FABRICATED_DOC_NAMES:
            if fake not in registry["document_names"]:
                assert fake not in answer

    def test_chat_corpus_meta_count_skips_search_and_matches_db(self):
        client = TestClient(app)
        mock_engine = MagicMock()
        mock_db = MagicMock()
        fake_registry = {
            "total_documents": 87,
            "pdf_documents": 82,
            "by_source_type": {"UNECE": 75, "FMVSS": 5, "Euro NCAP": 7},
            "document_names": ["UN_R94.pdf", "FMVSS_208_2024.pdf"],
        }
        app.dependency_overrides[get_search_engine] = lambda: mock_engine
        app.dependency_overrides[get_db] = lambda: mock_db
        with patch("registry.chat_intent.query_document_registry", return_value=fake_registry):
            with patch("registry.chat_intent.build_coverage_report", return_value={"summary": {}, "authorities": []}):
                try:
                    for transcript in (CORPUS_META_TRANSCRIPT_COUNT, CORPUS_META_TRANSCRIPT_COUNT_ALT):
                        resp = client.post("/api/v1/chat", json={"query": transcript, "top_k": 8})
                        assert resp.status_code == 200
                        data = resp.json()
                        assert data["gateway"].get("route") == "corpus_meta"
                        assert "87" in data["answer"]
                        assert "82" in data["answer"]
                        assert data["citations"] == []
                        for fake in FABRICATED_DOC_NAMES:
                            assert fake not in data["answer"]
                    assert mock_engine.search.call_count == 0
                finally:
                    app.dependency_overrides.pop(get_search_engine, None)
                    app.dependency_overrides.pop(get_db, None)

    def test_search_engine_corpus_meta_count_short_circuits(self, db_session):
        engine = RegulationSearchEngine()
        registry = query_document_registry(db_session)
        with patch.object(engine, "_dense_search_sqlite") as mock_dense:
            out = engine.search(db_session, CORPUS_META_TRANSCRIPT_COUNT, top_k=8, rerank=False)
            mock_dense.assert_not_called()
            assert out["metadata"]["response_route"] == "corpus_meta"
            assert str(registry["total_documents"]) in out["answer"]

    def test_chat_corpus_meta_uses_coverage_not_search(self):
        client = TestClient(app)
        mock_engine = MagicMock()
        mock_db = MagicMock()
        fake_report = {
            "summary": {
                "expected": 22,
                "ingested": 22,
                "complete_count": 19,
                "partial_count": 3,
                "missing": 0,
                "coverage_rate": 1.0,
                "completeness_rate": 0.864,
            },
            "authorities": [
                {
                    "authority": "UNECE",
                    "complete": ["UN_R94"],
                    "partial": ["UN_R14"],
                    "missing": [],
                }
            ],
        }
        app.dependency_overrides[get_search_engine] = lambda: mock_engine
        app.dependency_overrides[get_db] = lambda: mock_db
        with patch("registry.chat_intent.build_coverage_report", return_value=fake_report):
            try:
                resp = client.post(
                    "/api/v1/chat",
                    json={"query": CORPUS_META_TRANSCRIPT_REGS, "top_k": 3},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["gateway"].get("route") == "corpus_meta"
                assert "22" in data["answer"]
                assert "/api/v1/coverage" in data["answer"]
                mock_engine.search.assert_not_called()
            finally:
                app.dependency_overrides.pop(get_search_engine, None)
                app.dependency_overrides.pop(get_db, None)


class TestInjectionChatEndpoint:
    def test_exact_injection_transcript_blocked_without_llm(self):
        client = TestClient(app)
        mock_engine = MagicMock()
        app.dependency_overrides[get_search_engine] = lambda: mock_engine
        mock_db = MagicMock()
        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            resp = client.post("/api/v1/chat", json={"query": INJECTION_TRANSCRIPT_FR})
            assert resp.status_code == 200
            data = resp.json()
            assert data["gateway"].get("route") == "injection_blocked"
            low = data["answer"].lower()
            assert "cannot follow" in low or "refuse" in low
            assert "invent" in low or "citation" in low
            assert "[s1]" not in low and "[1]" not in low
            mock_engine.search.assert_not_called()
        finally:
            app.dependency_overrides.pop(get_search_engine, None)
            app.dependency_overrides.pop(get_db, None)

    def test_search_engine_short_circuits_injection(self, db_session):
        engine = RegulationSearchEngine()
        with patch.object(engine, "_dense_search_sqlite") as mock_dense:
            out = engine.search(db_session, INJECTION_TRANSCRIPT_FR, top_k=3, rerank=False)
            mock_dense.assert_not_called()
            assert out["metadata"]["response_route"] == "injection_blocked"
