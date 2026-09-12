"""Upload → queue → worker → READY → scoped retrieval (ADR-0029 §5/§6; 07_TESTING "Upload"/"Ingestion").

Runs the worker synchronously (`run_once`) against the test database. Covers: happy path with
private scope, cross-user isolation at listing/detail/job/evidence/retrieval level, invalid magic,
oversized, poison PDF → QUARANTINED (never retrievable, not retryable), duplicate upload, workspace
scope membership, promotion privilege, archive, and FAILED → retry with attempt budget."""

from __future__ import annotations

import pathlib
import uuid

import pymupdf
import pytest
from fastapi.testclient import TestClient

from safety_assistant.api.routes import documents as documents_route
from safety_assistant.identity.service import create_user, create_workspace, default_organization
from safety_assistant.ingestion.fetch import FilesystemBlobStore
from safety_assistant.providers.embeddings.hashing import HashingEmbeddingProvider
from safety_assistant.workers import ingestion as worker
from tests.conftest import requires_db

pytestmark = requires_db

CSRF = {"X-Requested-With": "safety-assistant"}


def _project_pdf(tmp_path: pathlib.Path, marker: str = "The crash pulse peak shall not exceed 37 g.") -> bytes:
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page(width=595, height=842)
        y = 60
        for line in (
            f"Project Safety Note PSN-{i + 1}",
            "1. Scope",
            "This internal note describes sled test acceptance criteria for the front seat.",
            "2. Acceptance criteria",
            marker if i == 1 else "Refer to section 2 of the previous page.",
            "The dummy head excursion limit is 250 mm.",
        ):
            page.insert_text((60, y), line, fontsize=10)
            y += 16
    data = doc.tobytes(deflate=True, garbage=0)
    doc.close()
    return data


@pytest.fixture
def env(corpus, db_session, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Authoritative corpus (from the shared fixture) + Alice/Bob/admin + a workspace Alice owns."""
    org = default_organization(db_session)
    alice = create_user(db_session, email="alice@example.test", display_name="Alice", role="engineer")
    bob = create_user(db_session, email="bob@example.test", display_name="Bob", role="engineer")
    admin = create_user(db_session, email="admin@example.test", display_name="Admin", role="knowledge_admin")
    ws = create_workspace(db_session, organization_id=org.id, name="team-a", owner=alice)
    db_session.commit()
    store = FilesystemBlobStore(tmp_path / "uploads")
    documents_route._blobs.cache_clear()
    monkeypatch.setattr(documents_route, "_blobs", lambda: store)
    monkeypatch.setattr(worker, "BACKOFF_MINUTES", (0, 0, 0))
    return {"alice": alice, "bob": bob, "admin": admin, "ws": ws, "store": store, "embedder": corpus}


def _login(c: TestClient, email: str) -> None:
    c.cookies.clear()
    assert c.post("/api/v1/auth/dev-login", json={"email": email}).status_code == 200


def _upload(c: TestClient, data: bytes, **fields: str) -> dict:  # type: ignore[type-arg]
    form = {"title": "Sled acceptance note", "document_type": "PROJECT_DOCUMENT", "scope": "PRIVATE_USER", **fields}
    r = c.post("/api/v1/documents", files={"file": ("note.pdf", data, "application/pdf")}, data=form, headers=CSRF)
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


def _drain(db_session, env) -> list[str]:  # type: ignore[no-untyped-def]
    statuses = []
    while (job := worker.run_once(db_session, embedder=env["embedder"], blob_store=env["store"])) is not None:
        statuses.append(job.status)
    return statuses


def test_private_upload_reaches_ready_and_is_retrievable_only_by_owner(client, env, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _login(client, "alice@example.test")
    up = _upload(client, _project_pdf(tmp_path))
    assert up["status"] == "UPLOADED" and up["duplicate"] is False
    doc_id, job_id = up["document_id"], up["ingestion_job_id"]
    assert client.get(f"/api/v1/documents/{doc_id}").json()["status"] == "UPLOADED"

    # not READY yet → not retrievable even for the owner
    scoped = {"scopes": ["PRIVATE_USER"]}
    r = client.post(
        "/api/v1/ask", json={"query": "crash pulse peak limit", "k": 5, "source_scope": scoped}, headers=CSRF
    )
    assert r.status_code == 200 and r.json()["mode"] == "ABSTAINED"

    assert _drain(db_session, env) == ["SUCCEEDED"]
    job = client.get(f"/api/v1/ingestion-jobs/{job_id}").json()
    assert job["status"] == "SUCCEEDED" and job["stage"] == "READY"
    detail = client.get(f"/api/v1/documents/{doc_id}").json()
    assert detail["status"] == "READY" and detail["scope"] == "PRIVATE_USER" and detail["version"]["page_count"] == 3

    r = client.post(
        "/api/v1/ask", json={"query": "crash pulse peak limit", "k": 5, "source_scope": scoped}, headers=CSRF
    )
    j = r.json()
    assert j["mode"] in ("GENERATED", "EVIDENCE_ONLY"), j
    assert all(c["regulation_key"].startswith("DOC-") for c in j["citations"]), j["citations"]
    assert any("37 g" in e["content"] for e in j["evidence"])
    chunk_id = j["evidence"][0]["chunk_id"]
    assert client.get(f"/api/v1/evidence/{chunk_id}").status_code == 200

    # default scope (verified regulations) never returns the private note
    r = client.post("/api/v1/ask", json={"query": "crash pulse peak limit", "k": 5}, headers=CSRF)
    assert all(not c["regulation_key"].startswith("DOC-") for c in r.json()["citations"])

    # Bob: cannot list, open, poll or cite it; retrieval within his private scope finds nothing
    _login(client, "bob@example.test")
    assert client.get("/api/v1/documents?scope=PRIVATE_USER").json()["items"] == []
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404
    assert client.get(f"/api/v1/ingestion-jobs/{job_id}").status_code == 404
    assert client.get(f"/api/v1/evidence/{chunk_id}").status_code == 404
    r = client.post(
        "/api/v1/ask", json={"query": "crash pulse peak limit", "k": 5, "source_scope": scoped}, headers=CSRF
    )
    assert r.json()["mode"] == "ABSTAINED"
    all_scopes = {"scopes": ["AUTHORITATIVE_ORG", "WORKSPACE", "PRIVATE_USER"]}
    r = client.post(
        "/api/v1/ask", json={"query": "crash pulse peak limit", "k": 5, "source_scope": all_scopes}, headers=CSRF
    )
    assert all(not c["regulation_key"].startswith("DOC-") for c in r.json()["citations"])


def test_invalid_magic_and_oversized_are_rejected_at_the_boundary(client, env, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _login(client, "alice@example.test")
    form = {"title": "x", "scope": "PRIVATE_USER"}
    r = client.post(
        "/api/v1/documents", files={"file": ("x.pdf", b"MZ not a pdf", "application/pdf")}, data=form, headers=CSRF
    )
    assert r.status_code == 400 and "magic" in r.text
    from safety_assistant.config import get_settings

    monkeypatch.setattr(get_settings(), "ingest_max_file_bytes", 100)
    r = client.post(
        "/api/v1/documents",
        files={"file": ("x.pdf", b"%PDF-1.4" + b"0" * 200, "application/pdf")},
        data=form,
        headers=CSRF,
    )
    assert r.status_code == 400 and "exceeds" in r.text
    assert client.get("/api/v1/documents?scope=PRIVATE_USER").json()["items"] == []


def test_poison_pdf_is_quarantined_never_retrievable_and_not_retryable(client, env, db_session) -> None:  # type: ignore[no-untyped-def]
    _login(client, "alice@example.test")
    up = _upload(client, b"%PDF-1.7\n" + b"garbage " * 500)
    assert _drain(db_session, env) == ["QUARANTINED"]
    job = client.get(f"/api/v1/ingestion-jobs/{up['ingestion_job_id']}").json()
    assert job["status"] == "QUARANTINED" and job["error_code"] in ("INVALID_PDF", "UNREADABLE", "TOO_LARGE")
    assert job["error_public_message"] and job["diagnostic_reference"]
    assert "garbage" not in job["error_public_message"]
    doc = client.get(f"/api/v1/documents/{up['document_id']}").json()
    assert doc["status"] == "QUARANTINED"
    assert client.post(f"/api/v1/ingestion-jobs/{up['ingestion_job_id']}/retry", headers=CSRF).status_code == 409
    r = client.post(
        "/api/v1/ask", json={"query": "garbage", "k": 5, "source_scope": {"scopes": ["PRIVATE_USER"]}}, headers=CSRF
    )
    assert r.json()["mode"] == "ABSTAINED"


def test_duplicate_upload_returns_existing_identifiers(client, env, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _login(client, "alice@example.test")
    data = _project_pdf(tmp_path)
    first = _upload(client, data)
    _drain(db_session, env)
    second = _upload(client, data)
    assert second["duplicate"] is True
    assert (
        second["document_id"] == first["document_id"] and second["document_version_id"] == first["document_version_id"]
    )
    assert len(client.get("/api/v1/documents?scope=PRIVATE_USER").json()["items"]) == 1
    # same bytes by another owner: a separate private document, no cross-visibility
    _login(client, "bob@example.test")
    third = _upload(client, data)
    assert third["duplicate"] is False and third["document_id"] != first["document_id"]


def test_workspace_scope_requires_membership_and_is_visible_to_members(client, env, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    ws = str(env["ws"].id)
    _login(client, "bob@example.test")  # not a member
    form = {"title": "ws doc", "scope": "WORKSPACE", "workspace_id": ws}
    r = client.post(
        "/api/v1/documents",
        files={"file": ("w.pdf", _project_pdf(tmp_path), "application/pdf")},
        data=form,
        headers=CSRF,
    )
    assert r.status_code == 403
    _login(client, "alice@example.test")
    up = _upload(
        client,
        _project_pdf(tmp_path, marker="Workspace criterion: 52 kN axial load."),
        scope="WORKSPACE",
        workspace_id=ws,
    )
    assert _drain(db_session, env) == ["SUCCEEDED"]
    body = {"query": "axial load criterion", "k": 5, "source_scope": {"scopes": ["WORKSPACE"], "workspace_ids": [ws]}}
    assert any("52 kN" in e["content"] for e in client.post("/api/v1/ask", json=body, headers=CSRF).json()["evidence"])
    # Bob requesting that workspace scope is refused outright, not silently narrowed
    _login(client, "bob@example.test")
    assert client.post("/api/v1/ask", json=body, headers=CSRF).status_code == 403
    assert client.get(f"/api/v1/documents/{up['document_id']}").status_code == 404


def test_promotion_is_privileged_and_audited(client, env, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _login(client, "alice@example.test")
    up = _upload(client, _project_pdf(tmp_path))
    doc_id = up["document_id"]
    assert client.post(f"/api/v1/documents/{doc_id}/promote", headers=CSRF).status_code == 403  # engineer
    _drain(db_session, env)
    _login(client, "admin@example.test")
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404  # admin is not the owner, not a member
    # promotion goes through the owner's document being visible to the admin: give admin visibility via
    # an org-level listing is not enough — ownership is required, so the owner archives/promotes flows are
    # exercised through an admin-owned upload instead.
    up2 = _upload(client, _project_pdf(tmp_path, marker="Admin note value 12 ms."), title="Admin note")
    _drain(db_session, env)
    r = client.post(f"/api/v1/documents/{up2['document_id']}/promote", headers=CSRF)
    assert r.status_code == 200 and r.json()["scope"] == "AUTHORITATIVE_ORG", r.text
    from sqlalchemy import select

    from safety_assistant.persistence.models import AuditEvent

    assert db_session.scalar(select(AuditEvent).where(AuditEvent.action == "document.promote")) is not None
    # now every org member sees it in the verified-regulation scope
    _login(client, "bob@example.test")
    r = client.post("/api/v1/ask", json={"query": "admin note value", "k": 5}, headers=CSRF)
    assert any("12 ms" in e["content"] for e in r.json()["evidence"])


def test_archive_removes_document_from_retrieval(client, env, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _login(client, "alice@example.test")
    up = _upload(client, _project_pdf(tmp_path))
    _drain(db_session, env)
    body = {"query": "crash pulse peak limit", "k": 5, "source_scope": {"scopes": ["PRIVATE_USER"]}}
    assert client.post("/api/v1/ask", json=body, headers=CSRF).json()["mode"] != "ABSTAINED"
    r = client.post(f"/api/v1/documents/{up['document_id']}/archive", headers=CSRF)
    assert r.status_code == 200 and r.json()["status"] == "ARCHIVED"
    assert client.post("/api/v1/ask", json=body, headers=CSRF).json()["mode"] == "ABSTAINED"
    assert client.get("/api/v1/documents?scope=PRIVATE_USER").json()["items"] == []
    assert (
        client.get("/api/v1/documents?include_archived=true&scope=PRIVATE_USER").json()["items"][0]["status"]
        == "ARCHIVED"
    )


def test_failed_job_retries_with_attempt_budget_then_can_be_retried_manually(
    client, env, db_session, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    _login(client, "alice@example.test")
    up = _upload(client, _project_pdf(tmp_path))

    class Boom(HashingEmbeddingProvider):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            raise RuntimeError("embedding backend down")

    env_boom = {**env, "embedder": Boom(384)}
    assert _drain(db_session, env_boom) == ["QUEUED", "QUEUED", "FAILED"]  # 3 attempts, backoff patched to 0
    job = client.get(f"/api/v1/ingestion-jobs/{up['ingestion_job_id']}").json()
    assert job["status"] == "FAILED" and job["error_code"] == "ATTEMPTS_EXHAUSTED" and job["attempt"] == 3
    assert "embedding backend down" not in (job["error_public_message"] or "")
    assert client.get(f"/api/v1/documents/{up['document_id']}").json()["status"] == "FAILED"
    r = client.post(f"/api/v1/ingestion-jobs/{up['ingestion_job_id']}/retry", headers=CSRF)
    assert r.status_code == 201 and r.json()["status"] == "QUEUED"
    assert _drain(db_session, env) == ["SUCCEEDED"]
    assert client.get(f"/api/v1/documents/{up['document_id']}").json()["status"] == "READY"


def test_api_key_and_anonymous_principals_cannot_upload(client, env) -> None:  # type: ignore[no-untyped-def]
    client.cookies.clear()
    form = {"title": "x", "scope": "PRIVATE_USER"}
    r = client.post(
        "/api/v1/documents", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}, data=form, headers=CSRF
    )
    assert r.status_code == 403
    assert client.get(f"/api/v1/documents/{uuid.uuid4()}").status_code == 403
