"""Browser session tokens, role → scope mapping, and production refusals (ADR-0029 §4/§6)."""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

from safety_assistant.api.dependencies.auth import ROLE_SCOPES
from safety_assistant.config import Settings
from safety_assistant.identity.service import issue_session, verify_session

SECRET = "x" * 40


def _settings(**kw: object) -> Settings:
    base: dict[str, object] = {"app_env": "test", "session_secret": SECRET, "session_ttl_hours": 1}
    base.update(kw)
    return Settings(**base)  # type: ignore[arg-type]


def test_session_round_trip() -> None:
    uid = uuid.uuid4()
    assert verify_session(issue_session(uid, _settings()), _settings()) == uid


def test_session_rejects_expired_wrong_secret_and_foreign_tokens() -> None:
    import jwt

    uid = uuid.uuid4()
    old = dt.datetime.now(dt.UTC) - dt.timedelta(hours=2)
    assert verify_session(issue_session(uid, _settings(), now=old), _settings()) is None
    assert verify_session(issue_session(uid, _settings()), _settings(session_secret="y" * 40)) is None
    access = jwt.encode({"sub": str(uid), "exp": 4102444800}, SECRET, algorithm="HS256")  # no typ=session
    assert verify_session(access, _settings()) is None
    assert verify_session("garbage", _settings()) is None
    assert verify_session(issue_session(uid, _settings()), _settings(session_secret="")) is None


def test_trd_roles_map_onto_scopes() -> None:
    assert "document:upload" in ROLE_SCOPES["engineer"]
    assert "confidential:query" in ROLE_SCOPES["engineer"]
    assert {"document:promote", "document:ingest", "audit:read"} <= ROLE_SCOPES["knowledge_admin"]
    assert "chat:query" not in ROLE_SCOPES["auditor"]
    assert ROLE_SCOPES["org_admin"] >= ROLE_SCOPES["knowledge_admin"] | ROLE_SCOPES["engineer"]


@pytest.mark.parametrize(
    "overrides",
    [{"dev_login_enabled": True}, {"session_secret": "short"}],
)
def test_production_refuses_dev_login_and_weak_session_secret(overrides: dict[str, object]) -> None:
    good: dict[str, object] = {
        "app_env": "production",
        "auth_mode": "api_key",
        "database_url": "postgresql+psycopg://u:strong@db/x",
        "session_secret": SECRET,
        "dev_login_enabled": False,
        "embedding_provider": "fastembed",
        "llm_provider": "openai_compatible",
    }
    Settings(**good)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refusing to start in production"):
        Settings(**{**good, **overrides})  # type: ignore[arg-type]
