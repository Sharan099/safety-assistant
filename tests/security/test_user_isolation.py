"""Cross-user isolation over HTTP (06_SECURITY "Multi-user isolation tests"):
User A cannot list, fetch, edit or post into User B's conversations — by guessed UUID or otherwise —
and a foreign id is indistinguishable from a missing one. API-key principals have no user identity."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from safety_assistant.identity.service import create_user
from tests.conftest import requires_db

pytestmark = requires_db

CSRF = {"X-Requested-With": "safety-assistant"}


@pytest.fixture
def client(clean_db, db_session):  # type: ignore[no-untyped-def]
    from safety_assistant.api.main import app

    create_user(db_session, email="a@example.test", display_name="A", role="engineer")
    create_user(db_session, email="b@example.test", display_name="B", role="engineer")
    create_user(db_session, email="audit@example.test", display_name="Aud", role="auditor")
    db_session.commit()
    return TestClient(app)


def _login(c: TestClient, email: str) -> None:
    c.cookies.clear()
    assert c.post("/api/v1/auth/dev-login", json={"email": email}, headers=CSRF).status_code == 200


def test_user_a_cannot_see_or_touch_user_b_conversations(client) -> None:  # type: ignore[no-untyped-def]
    _login(client, "b@example.test")
    b_conv = client.post("/api/v1/conversations", json={"title": "B private"}, headers=CSRF).json()["id"]

    _login(client, "a@example.test")
    assert client.get("/api/v1/conversations").json()["items"] == []
    assert client.get("/api/v1/conversations?q=private").json()["items"] == []
    assert client.get(f"/api/v1/conversations/{b_conv}").status_code == 404
    assert client.patch(f"/api/v1/conversations/{b_conv}", json={"title": "pwned"}, headers=CSRF).status_code == 404
    assert (
        client.post(f"/api/v1/conversations/{b_conv}/messages", json={"content": "hi"}, headers=CSRF).status_code == 404
    )
    # a missing id looks the same as a foreign one (no existence oracle)
    assert client.get(f"/api/v1/conversations/{uuid.uuid4()}").status_code == 404

    _login(client, "b@example.test")
    assert client.get(f"/api/v1/conversations/{b_conv}").json()["title"] == "B private"


def test_anonymous_and_api_key_principals_have_no_user_data(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client.cookies.clear()
    assert client.get("/api/v1/conversations").status_code == 403
    assert client.post("/api/v1/conversations", json={}, headers=CSRF).status_code == 403
    assert client.get("/api/v1/me").status_code == 403


def test_auditor_role_can_keep_history_but_cannot_query(client) -> None:  # type: ignore[no-untyped-def]
    _login(client, "audit@example.test")
    cid = client.post("/api/v1/conversations", json={}, headers=CSRF).json()["id"]
    r = client.post(f"/api/v1/conversations/{cid}/messages", json={"content": "limit?"}, headers=CSRF)
    assert r.status_code == 403 and "chat:query" in r.text


def test_session_cookie_cannot_be_reused_across_secrets(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.config import get_settings

    _login(client, "a@example.test")
    assert client.get("/api/v1/me").status_code == 200
    monkeypatch.setattr(get_settings(), "session_secret", "rotated-secret-rotated-secret-rotated-0")
    assert client.get("/api/v1/me").status_code == 401
