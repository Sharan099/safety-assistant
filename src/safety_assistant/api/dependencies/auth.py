"""Authentication + authorization dependencies (CLAUDE.md §14).

Modes (Settings.auth_mode):
- ``none``     development only: every request is an anonymous RegulationViewer.
               Production settings refuse this mode at startup.
- ``api_key``  ``Authorization: Bearer <key>``; the key maps to a role.
- ``oidc``     JWT bearer verified against the issuer's JWKS (M11).

Roles → scopes are fixed in code; authorization happens *before* retrieval by
requiring a scope on the route and by narrowing `ScopeFilter.data_classes`.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Request, status

from safety_assistant.config import Settings, get_settings

ROLE_SCOPES: dict[str, frozenset[str]] = {
    "RegulationViewer": frozenset({"regulation:read", "chat:query"}),
    "Engineer": frozenset({"regulation:read", "chat:query", "confidential:query"}),
    "DataIngestor": frozenset({"regulation:read", "chat:query", "document:upload", "document:ingest"}),
    "Auditor": frozenset({"regulation:read", "audit:read"}),
    "SafetyAdmin": frozenset(
        {
            "regulation:read",
            "chat:query",
            "confidential:query",
            "document:upload",
            "document:ingest",
            "index:rebuild",
            "audit:read",
            "system:admin",
        }
    ),  # fmt: skip
}


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    anonymous: bool = False

    @property
    def data_classes(self) -> tuple[str, ...]:
        return ("PUBLIC", "CONFIDENTIAL") if "confidential:query" in self.scopes else ("PUBLIC",)


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


def _from_api_key(token: str | None, settings: Settings) -> Principal:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    for key, role in settings.api_keys.items():
        if hmac.compare_digest(key, token):  # constant-time, no early exit on prefix
            if role not in ROLE_SCOPES:
                raise HTTPException(status.HTTP_403_FORBIDDEN, f"unknown role configured: {role}")
            return Principal(subject=f"apikey:{key[:4]}…", role=role, scopes=ROLE_SCOPES[role])
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")


def get_principal(request: Request, settings: Settings = Depends(get_settings)) -> Principal:
    if settings.auth_mode == "none":
        return Principal(
            subject="anonymous", role="RegulationViewer", scopes=ROLE_SCOPES["RegulationViewer"], anonymous=True
        )
    if settings.auth_mode == "api_key":
        return _from_api_key(_bearer(request), settings)
    from safety_assistant.api.dependencies.oidc import principal_from_jwt  # lazy: optional dependency

    return principal_from_jwt(_bearer(request), settings)


def require_scope(scope: str):  # type: ignore[no-untyped-def]
    def _check(principal: Principal = Depends(get_principal)) -> Principal:
        if scope not in principal.scopes:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"scope '{scope}' required")
        return principal

    return _check
