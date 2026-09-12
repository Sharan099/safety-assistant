"""Shared integration fixtures: synthetic two-version corpus (hashing embeddings), an
evidence-aware mock LLM, and a TestClient whose AnswerService can be swapped via `client.llm_holder`."""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from safety_assistant.ingestion.fetch import FilesystemBlobStore
from safety_assistant.ingestion.workflows import ingest_source
from safety_assistant.providers.embeddings.hashing import HashingEmbeddingProvider
from safety_assistant.providers.llm import LLMMessage, LLMResponse, LLMUnavailable
from tests.support.minireg import registry_for


class _EvidenceAwareMock:
    """Mock LLM that cites the first evidence block and copies its numbers — what a
    compliant model does. Lets the pipeline (gate → generate → validate) run end to end."""

    name = "mock"
    model = "mock-evidence-aware"

    def generate(self, messages: list[LLMMessage], *, schema=None, temperature=0.0, max_tokens=1024):  # type: ignore[no-untyped-def]
        user = messages[-1].content
        first = user.split("<evidence ", 2)[1]
        eid = first.split('"', 2)[1]
        body = first.split(">", 1)[1].split("</evidence>")[0]
        question = user.rsplit("<question>", 1)[1].lower()
        candidates = [ln for ln in body.splitlines() if "shall not exceed" in ln]
        line = next(
            (ln for ln in candidates if "thorax" in ln.lower() and "thorax" in question),
            candidates[0] if candidates else body.strip().splitlines()[-1],
        )
        payload = {
            "answer": f"{line.strip()} [{eid}]",
            "claims": [{"text": line.strip(), "evidence_ids": [eid], "kind": "REQUIREMENT"}],
            "warnings": [],
            "insufficient_evidence": False,
        }
        parsed = schema.model_validate(payload) if schema else None
        return LLMResponse(
            content=str(payload), model=self.model, provider=self.name, parsed=parsed, usage={"total_tokens": 42}
        )


class _FailingLLM:
    name = "mock"
    model = "down"

    def generate(self, messages, *, schema=None, temperature=0.0, max_tokens=1024):  # type: ignore[no-untyped-def]
        raise LLMUnavailable("provider 503")


@pytest.fixture
def corpus(clean_db, db_session, tmp_path: pathlib.Path):  # type: ignore[no-untyped-def]
    reg = registry_for(tmp_path)
    emb = HashingEmbeddingProvider(384)
    store = FilesystemBlobStore(tmp_path / "b")
    for key in ("test-un-r999-rev1", "test-un-r999-rev2"):
        assert (
            ingest_source(db_session, key, registry=reg, blob_store=store, embedder=emb, repo_root=tmp_path).status
            == "SUCCEEDED"
        )
    return emb


@pytest.fixture
def client(corpus, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from safety_assistant.api.routes import query as q
    from safety_assistant.generation.service import AnswerService
    from safety_assistant.retrieval import RetrievalService
    from safety_assistant.retrieval.service import RetrievalConfig

    retrieval = RetrievalService(embedder=corpus, config=RetrievalConfig(min_shared_terms=1))
    holder = {"llm": _EvidenceAwareMock()}
    q._retrieval.cache_clear()
    q._answers.cache_clear()
    monkeypatch.setattr(q, "_retrieval", lambda: retrieval)
    monkeypatch.setattr(q, "_answers", lambda: AnswerService(retrieval, llm=holder["llm"]))
    from safety_assistant.api.main import app

    c = TestClient(app)
    c.llm_holder = holder  # type: ignore[attr-defined]
    return c
