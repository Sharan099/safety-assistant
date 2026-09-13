"""OpenID Connect: discovery, JWKS, ID-token validation, and the authorization-code flow (PKCE).

Provider-neutral (Entra ID, Keycloak, Okta … anything serving `.well-known/openid-configuration`).
Nothing here touches the database; `api/routes/oidc_login.py` maps validated claims to users.
HTTP goes through an injectable client so tests run against an in-process fake provider.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

from safety_assistant.config import Settings

ALGORITHMS = ["RS256", "ES256"]
_DISCOVERY_TTL_S = 3600.0


class OidcError(Exception):
    """Configuration or protocol failure; the HTTP layer maps it to 4xx/5xx without leaking detail."""


@dataclass(frozen=True)
class Discovery:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str
    end_session_endpoint: str | None = None


_discovery_cache: dict[str, tuple[float, Discovery]] = {}
_jwks_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def discover(issuer: str, http: httpx.Client) -> Discovery:
    cached = _discovery_cache.get(issuer)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    resp = http.get(issuer.rstrip("/") + "/.well-known/openid-configuration", timeout=10)
    if resp.status_code != 200:
        raise OidcError(f"discovery failed ({resp.status_code})")
    doc = resp.json()
    try:
        disc = Discovery(
            issuer=str(doc["issuer"]),
            authorization_endpoint=str(doc["authorization_endpoint"]),
            token_endpoint=str(doc["token_endpoint"]),
            jwks_uri=str(doc["jwks_uri"]),
            end_session_endpoint=doc.get("end_session_endpoint"),
        )
    except KeyError as exc:
        raise OidcError(f"discovery document missing {exc}") from exc
    if disc.issuer.rstrip("/") != issuer.rstrip("/"):
        raise OidcError("discovery issuer mismatch")
    _discovery_cache[issuer] = (time.monotonic() + _DISCOVERY_TTL_S, disc)
    return disc


def _signing_key(token: str, jwks_uri: str, http: httpx.Client) -> Any:
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    cached = _jwks_cache.get(jwks_uri)
    keys = cached[1] if cached and cached[0] > time.monotonic() else None
    for attempt in range(2):
        if keys is None or attempt == 1:  # refresh once on an unknown kid (key rotation)
            resp = http.get(jwks_uri, timeout=10)
            if resp.status_code != 200:
                raise OidcError(f"jwks fetch failed ({resp.status_code})")
            keys = resp.json()
            _jwks_cache[jwks_uri] = (time.monotonic() + _DISCOVERY_TTL_S, keys)
        for jwk in keys.get("keys", []):
            if kid is None or jwk.get("kid") == kid:
                return jwt.PyJWK(jwk).key
    raise OidcError("no matching signing key")


def verify_token(
    token: str, *, issuer: str, audience: str, http: httpx.Client, nonce: str | None = None
) -> dict[str, Any]:
    """Signature (JWKS), issuer, audience, expiry, iat, sub — and the nonce for ID tokens."""
    disc = discover(issuer, http)
    key = _signing_key(token, disc.jwks_uri, http)
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            key,
            algorithms=ALGORITHMS,
            audience=audience,
            issuer=disc.issuer,
            options={"require": ["exp", "iat", "sub"]},
            leeway=30,
        )
    except jwt.PyJWTError as exc:
        raise OidcError(f"token rejected: {type(exc).__name__}") from exc
    if nonce is not None and claims.get("nonce") != nonce:
        raise OidcError("nonce mismatch")
    return claims


# ------------------------------------------------------------------ authorization-code flow


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


@dataclass(frozen=True)
class FlowState:
    state: str
    nonce: str
    code_verifier: str
    return_to: str

    @classmethod
    def new(cls, return_to: str) -> FlowState:
        return cls(
            state=secrets.token_urlsafe(24),
            nonce=secrets.token_urlsafe(24),
            code_verifier=_b64url(secrets.token_bytes(48)),
            return_to=return_to,
        )

    def code_challenge(self) -> str:
        return _b64url(hashlib.sha256(self.code_verifier.encode()).digest())

    def seal(self, secret: str, ttl_s: int = 600) -> str:
        payload = {
            "typ": "oidc_flow",
            "st": self.state,
            "n": self.nonce,
            "cv": self.code_verifier,
            "rt": self.return_to,
        }
        return jwt.encode({**payload, "exp": int(time.time()) + ttl_s}, secret, algorithm="HS256")

    @classmethod
    def unseal(cls, sealed: str, secret: str) -> FlowState:
        try:
            p = jwt.decode(sealed, secret, algorithms=["HS256"], options={"require": ["exp"]})
        except jwt.PyJWTError as exc:
            raise OidcError("login flow expired or invalid") from exc
        if p.get("typ") != "oidc_flow":
            raise OidcError("not a login-flow token")
        return cls(state=p["st"], nonce=p["n"], code_verifier=p["cv"], return_to=p["rt"])


def authorization_url(settings: Settings, flow: FlowState, http: httpx.Client) -> str:
    disc = discover(settings.oidc_issuer, http)
    params = {
        "response_type": "code",
        "client_id": settings.oidc_client_id,
        "redirect_uri": settings.oidc_redirect_uri,
        "scope": settings.oidc_scopes,
        "state": flow.state,
        "nonce": flow.nonce,
        "code_challenge": flow.code_challenge(),
        "code_challenge_method": "S256",
    }
    return str(httpx.URL(disc.authorization_endpoint).copy_merge_params(params))


def exchange_code(settings: Settings, flow: FlowState, code: str, http: httpx.Client) -> dict[str, Any]:
    """Authorization-code → tokens; returns the validated ID-token claims (never the raw tokens)."""
    disc = discover(settings.oidc_issuer, http)
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": settings.oidc_redirect_uri,
        "client_id": settings.oidc_client_id,
        "code_verifier": flow.code_verifier,
    }
    if settings.oidc_client_secret:
        resp = http.post(
            disc.token_endpoint, data=data, auth=(settings.oidc_client_id, settings.oidc_client_secret), timeout=15
        )
    else:
        resp = http.post(disc.token_endpoint, data=data, timeout=15)
    if resp.status_code != 200:
        raise OidcError(f"token exchange failed ({resp.status_code})")
    body = resp.json()
    id_token = body.get("id_token")
    if not id_token:
        raise OidcError("token response has no id_token")
    return verify_token(
        id_token, issuer=settings.oidc_issuer, audience=settings.oidc_client_id, http=http, nonce=flow.nonce
    )
