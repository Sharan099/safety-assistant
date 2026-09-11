"""OIDC bearer-token verification (auth_mode=oidc).

Verifies RS256/ES256 JWTs against the issuer's JWKS (fetched once, cached),
checks audience/issuer/expiry, and maps a `role`/`roles` claim to a Principal.
Requires the optional `PyJWT[crypto]` dependency (installed in the `auth` extra).
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException, status

from safety_assistant.api.dependencies.auth import ROLE_SCOPES, Principal
from safety_assistant.config import Settings

_ROLE_CLAIMS = ("role", "roles", "groups")


@lru_cache(maxsize=4)
def _jwk_client(issuer: str):  # type: ignore[no-untyped-def]
    import jwt

    return jwt.PyJWKClient(issuer.rstrip("/") + "/.well-known/jwks.json", cache_keys=True)


def principal_from_jwt(token: str | None, settings: Settings) -> Principal:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    if not settings.oidc_issuer or not settings.oidc_audience:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "OIDC not configured")
    try:
        import jwt

        key = _jwk_client(settings.oidc_issuer).get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256", "ES256"],
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer,
            options={"require": ["exp", "iat", "sub"]},
        )
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "PyJWT not installed") from exc
    except Exception as exc:  # noqa: BLE001 — any verification failure is a 401, details stay server-side
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
