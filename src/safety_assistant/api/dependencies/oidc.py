"""OIDC bearer-token verification (auth_mode=oidc): access tokens presented by machines and by SPAs
that hold their own tokens. Discovery, JWKS caching and validation live in identity/oidc.py; this
module only maps validated claims to a role-based Principal."""

from __future__ import annotations

from fastapi import HTTPException, status

from safety_assistant.api.dependencies.auth import ROLE_SCOPES, Principal
from safety_assistant.config import Settings

_ROLE_CLAIMS = ("role", "roles", "groups")


def principal_from_jwt(token: str | None, settings: Settings) -> Principal:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    audience = settings.oidc_audience or settings.oidc_client_id
    if not settings.oidc_issuer or not audience:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "OIDC not configured")
    from safety_assistant.api.routes.oidc_login import http_client
    from safety_assistant.identity.oidc import OidcError, verify_token

    try:
        claims = verify_token(token, issuer=settings.oidc_issuer, audience=audience, http=http_client())
    except OidcError as exc:  # any verification failure is a 401; the reason stays server-side
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc
    except Exception as exc:  # noqa: BLE001 — provider unreachable etc.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc
    role = "RegulationViewer"
    for claim in _ROLE_CLAIMS:
        value = claims.get(claim)
        candidates = value if isinstance(value, list) else [value]
        for c in candidates:
            if c in ROLE_SCOPES:
                role = c
                break
    return Principal(subject=str(claims["sub"]), role=role, scopes=ROLE_SCOPES[role])
