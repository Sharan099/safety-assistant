"""Self-service email/password sign-up and sign-in over HTTP: hashing, generic failure messages
(no account-existence oracle at sign-in), per-account lockout, and CSRF on the login/signup POSTs
themselves (a plain cross-site form cannot add the header, so it cannot sign a victim into an
attacker-controlled account — "login CSRF")."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from safety_assistant.persistence.models import AuditEvent, User
from tests.conftest import requires_db

pytestmark = requires_db

CSRF = {"X-Requested-With": "safety-assistant"}


@pytest.fixture
def client(clean_db, db_session):  # type: ignore[no-untyped-def]
    from safety_assistant.api.main import app
    from safety_assistant.api.middleware.ratelimit import reset_auth_rate_limit

    reset_auth_rate_limit()
    return TestClient(app)


def test_signup_creates_an_engineer_and_signs_them_in(client, db_session) -> None:  # type: ignore[no-untyped-def]
    r = client.post(
        "/api/v1/auth/signup",
        json={
            "email": "new.engineer@example.test",
            "display_name": "New Engineer",
            "password": "correct-horse-battery",
        },
        headers=CSRF,
    )
    assert r.status_code == 201, r.text
    assert "session" in client.cookies
    assert "httponly" in r.headers["set-cookie"].lower()

    me = client.get("/api/v1/me").json()
    assert me["user"]["email"] == "new.engineer@example.test"
    assert me["user"]["roles"] == ["engineer"]
    assert "document:upload" in me["user"]["scopes"]

    user = db_session.scalar(select(User).where(User.email == "new.engineer@example.test"))
    assert user is not None
    assert user.password_hash is not None
    assert user.password_hash.startswith("scrypt$")
    assert "correct-horse-battery" not in user.password_hash  # never stored in the clear
    assert db_session.scalar(select(AuditEvent).where(AuditEvent.action == "auth.signup")) is not None


def test_signup_rejects_duplicate_email_short_password_and_password_containing_email(client) -> None:  # type: ignore[no-untyped-def]
    body = {"email": "dup@example.test", "display_name": "Dup", "password": "first-password-123"}
    assert client.post("/api/v1/auth/signup", json=body, headers=CSRF).status_code == 201
    assert client.post("/api/v1/auth/signup", json=body, headers=CSRF).status_code == 409

    short = {"email": "short@example.test", "display_name": "S", "password": "abc123"}
    assert client.post("/api/v1/auth/signup", json=short, headers=CSRF).status_code == 422

    containing = {"email": "bob@example.test", "display_name": "Bob", "password": "bob@example.testXYZ"}
    assert client.post("/api/v1/auth/signup", json=containing, headers=CSRF).status_code == 422


def test_signup_and_login_require_the_csrf_header(client) -> None:  # type: ignore[no-untyped-def]
    body = {"email": "nocsrf@example.test", "display_name": "N", "password": "long-enough-password"}
    assert client.post("/api/v1/auth/signup", json=body).status_code == 403
    client.post("/api/v1/auth/signup", json=body, headers=CSRF)
    login_body = {"email": body["email"], "password": body["password"]}
    assert client.post("/api/v1/auth/login", json=login_body).status_code == 403


def test_login_succeeds_with_correct_password_and_fails_generically_otherwise(client, db_session) -> None:  # type: ignore[no-untyped-def]
    signup = {"email": "signin@example.test", "display_name": "S", "password": "a-strong-password-1"}
    client.post("/api/v1/auth/signup", json=signup, headers=CSRF)
    client.cookies.clear()

    login_body = {"email": signup["email"], "password": signup["password"]}
    ok = client.post("/api/v1/auth/login", json=login_body, headers=CSRF)
    assert ok.status_code == 200 and "session" in client.cookies
    client.cookies.clear()

    wrong_pw_body = {"email": signup["email"], "password": "totally-wrong"}
    wrong_pw = client.post("/api/v1/auth/login", json=wrong_pw_body, headers=CSRF)
    unknown = client.post(
        "/api/v1/auth/login", json={"email": "nobody-at-all@example.test", "password": "whatever12345"}, headers=CSRF
    )
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json()["detail"] == unknown.json()["detail"] == "invalid email or password"
    assert "session" not in client.cookies

    failure_events = db_session.scalars(select(AuditEvent).where(AuditEvent.action == "auth.login_failed")).all()
    reasons = {e.metadata_["reason"] for e in failure_events if e.metadata_}
    assert reasons == {"bad_password", "no_such_account"}


def test_account_locks_after_repeated_failures_and_clears_on_success(client, db_session) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.config import get_settings

    signup = {"email": "lockout@example.test", "display_name": "L", "password": "the-real-password-99"}
    client.post("/api/v1/auth/signup", json=signup, headers=CSRF)
    client.cookies.clear()

    max_attempts = get_settings().login_max_attempts
    for _ in range(max_attempts):
        r = client.post("/api/v1/auth/login", json={"email": signup["email"], "password": "nope"}, headers=CSRF)
        assert r.status_code == 401

    user = db_session.scalar(select(User).where(User.email == signup["email"]))
    assert user.locked_until is not None

    # even the correct password is refused while locked — the account, not the password, gates it
    still_locked = client.post(
        "/api/v1/auth/login", json={"email": signup["email"], "password": signup["password"]}, headers=CSRF
    )
    assert still_locked.status_code == 401

    db_session.refresh(user)
    user.locked_until = None
    db_session.commit()
    recovered = client.post(
        "/api/v1/auth/login", json={"email": signup["email"], "password": signup["password"]}, headers=CSRF
    )
    assert recovered.status_code == 200
    db_session.refresh(user)
    assert user.failed_login_attempts == 0 and user.locked_until is None


def test_password_auth_can_be_disabled(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.config import get_settings

    monkeypatch.setattr(get_settings(), "password_auth_enabled", False)
    body = {"email": "off@example.test", "display_name": "Off", "password": "long-enough-password"}
    assert client.post("/api/v1/auth/signup", json=body, headers=CSRF).status_code == 404
    off_body = {"email": "x@example.test", "password": "y"}
    assert client.post("/api/v1/auth/login", json=off_body, headers=CSRF).status_code == 404
