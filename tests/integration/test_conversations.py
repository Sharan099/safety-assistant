"""Identity + conversations over HTTP (ADR-0029 §6): dev-login cookie, /me, preferences,
conversation CRUD, message persistence with citations, and conversation context reaching
the model as data (never as evidence)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from safety_assistant.identity.service import create_user, create_workspace, default_organization
from safety_assistant.persistence.models import AuditEvent, MessageCitation
from tests.conftest import requires_db
from tests.integration.conftest import _EvidenceAwareMock

pytestmark = requires_db

CSRF = {"X-Requested-With": "safety-assistant"}


@pytest.fixture
def users(corpus, db_session):  # type: ignore[no-untyped-def]
    org = default_organization(db_session)
    a = create_user(db_session, email="a@example.test", display_name="Alice", role="engineer")
    b = create_user(db_session, email="b@example.test", display_name="Bob", role="engineer")
    ws = create_workspace(db_session, organization_id=org.id, name="crash-team", owner=a)
    db_session.commit()
    return {"a": a, "b": b, "ws": ws}


def login(c: TestClient, email: str) -> None:
    c.cookies.clear()
    r = c.post("/api/v1/auth/dev-login", json={"email": email})
    assert r.status_code == 200, r.text
    assert "session" in c.cookies


def test_dev_login_sets_httponly_cookie_and_me_reflects_membership(client, users, db_session) -> None:  # type: ignore[no-untyped-def]
    r = client.post("/api/v1/auth/dev-login", json={"email": "a@example.test"})
    assert r.status_code == 200
    assert "httponly" in r.headers["set-cookie"].lower() and "samesite=lax" in r.headers["set-cookie"].lower()
    me = client.get("/api/v1/me").json()
    assert me["user"]["email"] == "a@example.test" and me["user"]["roles"] == ["engineer"]
    assert "document:upload" in me["user"]["scopes"]
    assert [w["name"] for w in me["workspaces"]] == ["crash-team"]
    assert db_session.scalar(select(AuditEvent).where(AuditEvent.action == "auth.dev_login")) is not None


def test_unknown_user_and_missing_cookie_are_rejected(client, users) -> None:  # type: ignore[no-untyped-def]
    assert client.post("/api/v1/auth/dev-login", json={"email": "nobody@example.test"}).status_code == 401
    client.cookies.clear()
    assert client.get("/api/v1/me").status_code == 403  # anonymous viewer has no user identity
    client.cookies.set("session", "forged")
    assert client.get("/api/v1/me").status_code == 401


def test_mutations_with_cookie_require_csrf_header(client, users) -> None:  # type: ignore[no-untyped-def]
    login(client, "a@example.test")
    assert client.post("/api/v1/conversations", json={}).status_code == 403
    assert client.post("/api/v1/conversations", json={}, headers=CSRF).status_code == 201


def test_preferences_patch_validates_workspace(client, users) -> None:  # type: ignore[no-untyped-def]
    login(client, "b@example.test")
    ws = str(users["ws"].id)
    assert client.patch("/api/v1/me/preferences", json={"default_workspace_id": ws}, headers=CSRF).status_code == 403
    r = client.patch("/api/v1/me/preferences", json={"answer_density": "concise", "ui_theme": "dark"}, headers=CSRF)
    assert r.status_code == 200 and r.json()["preferences"]["answer_density"] == "concise"
    assert client.get("/api/v1/me").json()["preferences"]["ui_theme"] == "dark"


def test_conversation_lifecycle_persists_messages_and_citations(client, users, db_session) -> None:  # type: ignore[no-untyped-def]
    login(client, "a@example.test")
    r = client.post("/api/v1/conversations", json={"source_scope": {"scopes": ["AUTHORITATIVE_ORG"]}}, headers=CSRF)
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    assert r.json()["title"] == "New investigation" and r.json()["title_locked"] is False

    r = client.post(
        f"/api/v1/conversations/{cid}/messages",
        json={"content": "What is the thorax compression criterion limit in R999?", "k": 4},
        headers=CSRF,
    )
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["answer"]["mode"] == "GENERATED"
    assert j["assistant_message"]["answer_mode"] == "GENERATED"
    assert j["assistant_message"]["citations"], "citations must persist with the assistant message"
    cite = j["assistant_message"]["citations"][0]
    assert cite["regulation_key"] == "UN-R999" and cite["evidence_available"] is True and cite["chunk_id"]
    assert j["conversation"]["title"].startswith("What is the thorax")  # generated once, not locked

    # restore: list + get return the thread with messages and citations in order
    listing = client.get("/api/v1/conversations").json()["items"]
    assert [c["id"] for c in listing] == [cid]
    full = client.get(f"/api/v1/conversations/{cid}").json()
    assert [m["role"] for m in full["messages"]] == ["user", "assistant"]
    assert full["messages"][1]["citations"][0]["label"] == cite["label"]
    assert db_session.scalar(select(MessageCitation)) is not None

    # user-edited title is locked; archive hides from default list and blocks new messages
    r = client.patch(f"/api/v1/conversations/{cid}", json={"title": "Thorax limits"}, headers=CSRF)
    assert r.json()["title"] == "Thorax limits" and r.json()["title_locked"] is True
    client.patch(f"/api/v1/conversations/{cid}", json={"archived": True}, headers=CSRF)
    assert client.get("/api/v1/conversations").json()["items"] == []
    assert client.get("/api/v1/conversations?archived=true").json()["items"][0]["id"] == cid
    assert (
        client.post(f"/api/v1/conversations/{cid}/messages", json={"content": "again?"}, headers=CSRF).status_code
        == 409
    )


def test_conversation_context_is_passed_as_data_not_evidence(client, users) -> None:  # type: ignore[no-untyped-def]
    seen: list[str] = []
    inner = _EvidenceAwareMock()

    class Spy:
        name, model = inner.name, inner.model

        def generate(self, messages, **kw):  # type: ignore[no-untyped-def]
            seen.append(messages[-1].content)
            return inner.generate(messages, **kw)

    client.llm_holder["llm"] = Spy()
    login(client, "a@example.test")
    cid = client.post("/api/v1/conversations", json={}, headers=CSRF).json()["id"]
    q = {"content": "What is the thorax compression criterion limit in R999?", "k": 4}
    assert client.post(f"/api/v1/conversations/{cid}/messages", json=q, headers=CSRF).status_code == 201
    assert "<conversation_context>" not in seen[0]
    q2 = {"content": "And the thorax limit again, same regulation?", "k": 4}
    r = client.post(f"/api/v1/conversations/{cid}/messages", json=q2, headers=CSRF)
    assert r.status_code == 201, r.text
    ctx = seen[1].split("<conversation_context>", 1)[1].split("</conversation_context>", 1)[0]
    assert "user: What is the thorax" in ctx and "assistant:" in ctx
    assert seen[1].index("<conversation_context>") < seen[1].index("<question>")
    # citations still resolve only to evidence retrieved for *this* request
    assert all(c["evidence_available"] for c in r.json()["assistant_message"]["citations"])


def test_workspace_scope_outside_membership_is_refused(client, users) -> None:  # type: ignore[no-untyped-def]
    login(client, "b@example.test")  # Bob is not in crash-team
    ws = str(users["ws"].id)
    body = {"source_scope": {"scopes": ["WORKSPACE"], "workspace_ids": [ws]}}
    assert client.post("/api/v1/conversations", json=body, headers=CSRF).status_code == 403
    assert client.post("/api/v1/conversations", json={"workspace_id": ws}, headers=CSRF).status_code == 403
    login(client, "a@example.test")
    assert client.post("/api/v1/conversations", json=body, headers=CSRF).status_code == 201
