"""Browser OIDC (authorization code + PKCE) against an in-process fake provider: state/nonce/PKCE,
issuer/audience/signature/expiry validation, user mapping and membership policy, open-redirect guard,
and the bearer path sharing the same verifier."""

from __future__ import annotations

import json
import time
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from safety_assistant.api.routes import oidc_login
from safety_assistant.config import get_settings
from safety_assistant.identity import oidc as oidc_mod
from tests.conftest import requires_db

pytestmark = requires_db

ISSUER = "https://idp.example.test/realms/safety"
CLIENT_ID = "safety-assistant-web"


class FakeIdp:
    """Enough of an OpenID provider to drive the flow: discovery, authorize (not called), token, jwks."""

    def __init__(self) -> None:
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = "kid-1"
        self.issued: list[dict[str, object]] = []
        self.claims: dict[str, object] = {"sub": "sub-alice", "email": "alice.oidc@example.test", "name": "Alice OIDC"}
        self.audience = CLIENT_ID
        self.code = "auth-code-1"
        self.last_token_request: dict[str, list[str]] | None = None

    def id_token(self, nonce: str, *, exp_offset: int = 300) -> str:
        now = int(time.time())
        payload = {
            **self.claims,
            "iss": ISSUER,
            "aud": self.audience,
            "iat": now,
            "exp": now + exp_offset,
            "nonce": nonce,
        }
        return jwt.encode(payload, self.key, algorithm="RS256", headers={"kid": self.kid})

    def jwks(self) -> dict[str, object]:
        pub = self.key.public_key().public_numbers()
        import base64

        def b64(n: int) -> str:
            return base64.urlsafe_b64encode(n.to_bytes((n.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()

        return {
            "keys": [{"kty": "RSA", "kid": self.kid, "use": "sig", "alg": "RS256", "n": b64(pub.n), "e": b64(pub.e)}]
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/.well-known/openid-configuration"):
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": ISSUER + "/protocol/openid-connect/auth",
                    "token_endpoint": ISSUER + "/protocol/openid-connect/token",
                    "jwks_uri": ISSUER + "/protocol/openid-connect/certs",
                },
            )
        if path.endswith("/certs"):
            return httpx.Response(200, json=self.jwks())
        if path.endswith("/token"):
            form = parse_qs(request.content.decode())
            self.last_token_request = form
            if form.get("code") != [self.code] or "code_verifier" not in form:
                return httpx.Response(400, json={"error": "invalid_grant"})
            nonce = str(form.get("nonce", [""])[0]) or self._pending_nonce
            return httpx.Response(
                200, json={"id_token": self.id_token(nonce), "access_token": "opaque", "token_type": "Bearer"}
            )
        return httpx.Response(404)

    _pending_nonce = ""

    def public_key_pem(self) -> bytes:
        return self.key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )


@pytest.fixture
def idp(monkeypatch: pytest.MonkeyPatch, clean_db):  # type: ignore[no-untyped-def]
    provider = FakeIdp()
    client = httpx.Client(transport=httpx.MockTransport(provider.handler))
    oidc_login.http_client.cache_clear()
    monkeypatch.setattr(oidc_login, "http_client", lambda: client)
    oidc_mod._discovery_cache.clear()
    oidc_mod._jwks_cache.clear()
    s = get_settings()
    for k, v in {
        "oidc_issuer": ISSUER,
        "oidc_client_id": CLIENT_ID,
        "oidc_redirect_uri": "http://localhost:3010/api/v1/auth/oidc/callback",
        "oidc_default_role": "engineer",
        "frontend_url": "http://localhost:3010",
    }.items():
        monkeypatch.setattr(s, k, v)
    return provider


def _start_login(client: TestClient, **params: str) -> tuple[str, str]:
    r = client.get("/api/v1/auth/oidc/login", params=params, follow_redirects=False)
    assert r.status_code == 302, r.text
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q["code_challenge_method"] == ["S256"] and q["response_type"] == ["code"] and q["client_id"] == [CLIENT_ID]
    assert "oidc_flow" in r.cookies and "HttpOnly" in r.headers["set-cookie"]
    return q["state"][0], q["nonce"][0]


def test_full_login_creates_user_membership_session_and_redirects(idp: FakeIdp) -> None:
    from safety_assistant.api.main import app

    client = TestClient(app)
    state, nonce = _start_login(client, return_to="/app/documents")
    idp._pending_nonce = nonce
    r = client.get("/api/v1/auth/oidc/callback", params={"code": idp.code, "state": state}, follow_redirects=False)
    assert r.status_code == 302, r.text
    assert r.headers["location"] == "http://localhost:3010/app/documents"
    assert "session" in client.cookies
    assert idp.last_token_request is not None and "code_verifier" in idp.last_token_request  # PKCE was sent
    me = client.get("/api/v1/me").json()
    assert me["user"]["email"] == "alice.oidc@example.test" and me["user"]["roles"] == ["engineer"]
    assert client.get("/api/v1/auth/oidc/methods").json()["oidc"] is True


def test_state_mismatch_nonce_mismatch_wrong_audience_and_expiry_are_rejected(idp: FakeIdp) -> None:
    from safety_assistant.api.main import app

    client = TestClient(app)
    state, nonce = _start_login(client)
    idp._pending_nonce = nonce
    assert (
        client.get(
            "/api/v1/auth/oidc/callback", params={"code": idp.code, "state": "forged"}, follow_redirects=False
        ).status_code
        == 401
    )
    idp._pending_nonce = "other-nonce"
    assert (
        client.get(
            "/api/v1/auth/oidc/callback", params={"code": idp.code, "state": state}, follow_redirects=False
        ).status_code
        == 401
    )
    idp._pending_nonce = nonce
    idp.audience = "another-client"
    assert (
        client.get(
            "/api/v1/auth/oidc/callback", params={"code": idp.code, "state": state}, follow_redirects=False
        ).status_code
        == 401
    )
    idp.audience = CLIENT_ID
    original = idp.id_token
    idp.id_token = lambda n, exp_offset=300: original(n, exp_offset=-600)  # type: ignore[method-assign]
    assert (
        client.get(
            "/api/v1/auth/oidc/callback", params={"code": idp.code, "state": state}, follow_redirects=False
        ).status_code
        == 401
    )
    assert "session" not in client.cookies
    # no flow cookie at all → 400, and the provider's own error → 401 without a session
    fresh = TestClient(app)
    assert (
        fresh.get("/api/v1/auth/oidc/callback", params={"code": "x", "state": "y"}, follow_redirects=False).status_code
        == 400
    )
    assert (
        fresh.get("/api/v1/auth/oidc/callback", params={"error": "access_denied"}, follow_redirects=False).status_code
        == 401
    )


def test_open_redirect_is_refused_and_membership_is_not_implied_by_authentication(
    idp: FakeIdp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from safety_assistant.api.main import app

    client = TestClient(app)
    state, nonce = _start_login(client, return_to="https://evil.example/phish")
    idp._pending_nonce = nonce
    monkeypatch.setattr(get_settings(), "oidc_default_role", "")
    r = client.get("/api/v1/auth/oidc/callback", params={"code": idp.code, "state": state}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "http://localhost:3010/app/home"
    # authenticated, but without a membership every per-user route is denied
    assert client.get("/api/v1/me").status_code == 403


def test_bearer_access_tokens_use_the_same_verifier(idp: FakeIdp, monkeypatch: pytest.MonkeyPatch) -> None:
    from safety_assistant.api.main import app

    monkeypatch.setattr(get_settings(), "auth_mode", "oidc")
    monkeypatch.setattr(get_settings(), "oidc_audience", CLIENT_ID)
    client = TestClient(app)
    token = idp.id_token("n/a")
    assert client.get("/api/v1/regulations", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    tampered = token[:-4] + "AAAA"
    assert client.get("/api/v1/regulations", headers={"Authorization": f"Bearer {tampered}"}).status_code == 401
    assert client.get("/api/v1/regulations").status_code == 401
    assert json.loads(client.get("/api/v1/auth/oidc/methods").text)["oidc"] is True
