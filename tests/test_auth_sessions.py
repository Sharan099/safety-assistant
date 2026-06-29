"""Session auth and cross-user confidential isolation tests."""

from __future__ import annotations

import datetime

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from database.models import Test, TestAuditLog, User
from registry.auth import hash_password
from registry.harness_security import check_harness_access


@pytest.fixture
def client(db_session, two_users):
    """TestClient with get_db overridden to in-memory session (users pre-seeded)."""
    from database.connection import get_db

    def _override():
        yield db_session

    app.dependency_overrides[get_db] = _override
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def two_users(db_session):
    db_session.add_all(
        [
            User(user_id="user_a", username="alice", password_hash=hash_password("pass-a")),
            User(user_id="user_b", username="bob", password_hash=hash_password("pass-b")),
        ]
    )
    db_session.commit()
    return {"a": "alice", "b": "bob"}


def _login(client: TestClient, username: str, password: str) -> None:
    res = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200, res.text


def test_login_logout_and_me(client, two_users):
    unauth = client.get("/api/v1/auth/me")
    assert unauth.status_code == 401

    bad = client.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong"})
    assert bad.status_code == 401

    _login(client, "alice", "pass-a")
    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["user_id"] == "user_a"

    out = client.post("/api/v1/auth/logout")
    assert out.status_code == 200
    assert client.get("/api/v1/auth/me").status_code == 401


def test_harness_query_requires_session(db_session, two_users):
    db_session.add(
        Test(
            test_id="TEST-SEC-ISO",
            program="P",
            date=datetime.date(2026, 6, 28),
            test_type="PHYSICAL_CRASH",
            impact_mode="FRONTAL_OFFSET",
            confidential_tier=True,
            owner_user_id="user_a",
        )
    )
    db_session.commit()

    from fastapi.testclient import TestClient
    from database.connection import get_db

    def _override():
        yield db_session

    app.dependency_overrides[get_db] = _override
    client = TestClient(app)
    try:
        res = client.post(
            "/api/v1/harness/query",
            json={"query": "TEST-2026-06-001 THCC FRONTAL", "model_key": "ollama", "model_id": "llama3"},
        )
        assert res.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_cross_user_confidential_isolation(db_session, two_users):
    db_session.add(
        Test(
            test_id="TEST-OWN-A",
            program="P",
            date=datetime.date(2026, 6, 28),
            test_type="PHYSICAL_CRASH",
            impact_mode="FRONTAL_OFFSET",
            confidential_tier=True,
            owner_user_id="user_a",
        )
    )
    db_session.commit()

    check_harness_access(db_session, "ollama", "llama3", "user_a", ["TEST-OWN-A"])
    audit_a = (
        db_session.query(TestAuditLog)
        .filter(TestAuditLog.resource == "TEST-OWN-A", TestAuditLog.user_id == "user_a")
        .first()
    )
    assert audit_a is not None
    assert "AUTHORIZED" in audit_a.details

    with pytest.raises(HTTPException) as exc:
        check_harness_access(db_session, "ollama", "llama3", "user_b", ["TEST-OWN-A"])
    assert exc.value.status_code == 403

    audit_b = (
        db_session.query(TestAuditLog)
        .filter(TestAuditLog.resource == "TEST-OWN-A", TestAuditLog.user_id == "user_b")
        .first()
    )
    assert audit_b is not None
    assert "OWNER_DENIED" in audit_b.details


def test_harness_query_owner_sees_own_data(client, db_session, two_users):
    db_session.add(
        Test(
            test_id="TEST-2026-06-001",
            program="P",
            date=datetime.date(2026, 6, 28),
            test_type="PHYSICAL_CRASH",
            impact_mode="FRONTAL_OFFSET",
            confidential_tier=True,
            owner_user_id="user_a",
        )
    )
    db_session.commit()
    from database.models import TestResult

    db_session.add(
        TestResult(
            test_id="TEST-2026-06-001",
            channel="CH1",
            injury_criterion="ThCC",
            value=38.4,
            pass_fail="PASS",
            linked_regulation_clause="UN_R94#5.2.1.4",
        )
    )
    db_session.commit()

    _login(client, "alice", "pass-a")
    res = client.post(
        "/api/v1/harness/query",
        json={"query": "TEST-2026-06-001 THCC FRONTAL", "model_key": "ollama", "model_id": "llama3"},
    )
    assert res.status_code == 200
    assert res.json()["results"]

    client.post("/api/v1/auth/logout")
    _login(client, "bob", "pass-b")
    res_b = client.post(
        "/api/v1/harness/query",
        json={"query": "TEST-2026-06-001 THCC FRONTAL", "model_key": "ollama", "model_id": "llama3"},
    )
    assert res_b.status_code == 200
    assert res_b.json()["results"] == []
