"""Authentication + authorization dependencies (CLAUDE.md, ADR-0029 §4).

Bearer modes (Settings.auth_mode):
- ``none``     development only: every request is an anonymous RegulationViewer.
               Production settings refuse this mode at startup.
- ``api_key``  ``Authorization: Bearer <key>``; the key maps to a role (scripts/CI; no user row).
- ``oidc``     JWT bearer verified against the issuer's JWKS; the subject is resolved to a
               persisted user + memberships when one exists.

Browser sessions: an HttpOnly ``session`` cookie (identity.service.issue_session) resolves to a
persisted user regardless of auth_mode. Cookie-authenticated unsafe requests must carry
``X-Requested-With`` (CSRF).

Roles → scopes are fixed in code; authorization happens *before* retrieval by
requiring a scope on the route and by narrowing `ScopeFilter`.
"""

from __future__ import annotations

import hmac
import uuid
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from safety_assistant.config import Settings, get_settings
from safety_assistant.persistence import get_session

SESSION_COOKIE = "session"
CSRF_HEADER = "x-requested-with"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

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
# Organization membership roles (TRD) on top of the legacy API-key roles above.
ROLE_SCOPES["engineer"] = ROLE_SCOPES["Engineer"] | {"document:upload"}
ROLE_SCOPES["knowledge_admin"] = ROLE_SCOPES["DataIngestor"] | {"confidential:query", "document:promote", "audit:read"}
ROLE_SCOPES["auditor"] = ROLE_SCOPES["Auditor"]
ROLE_SCOPES["org_admin"] = ROLE_SCOPES["SafetyAdmin"] | {"document:promote"}


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    anonymous: bool = False
    # Persisted identity (None for API keys / anonymous). Retrieval scope and every
    # conversation/document query are built from these, never from client input.
    user_id: uuid.UUID | None = None
    organization_ids: tuple[uuid.UUID, ...] = ()
    workspace_ids: tuple[uuid.UUID, ...] = ()

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


def principal_for_user(session: Session, user_id: uuid.UUID) -> Principal | None:
    """Persisted user → Principal. Scopes are the union over the user's organization roles."""
    from safety_assistant.identity.service import load_identity

    ident = load_identity(session, user_id)
    if ident is None or not ident.memberships:
        return None
    scopes: frozenset[str] = frozenset()
    for role in ident.roles:
        scopes |= ROLE_SCOPES.get(role, frozenset())
    return Principal(
        subject=f"user:{ident.user.id}",
        role="+".join(ident.roles),
        scopes=scopes,
        user_id=ident.user.id,
        organization_ids=ident.organization_ids,
        workspace_ids=ident.workspace_ids,
    )


def _from_cookie(request: Request, settings: Settings, session: Session) -> Principal | None:
    from safety_assistant.identity.service import verify_session

    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    user_id = verify_session(token, settings)
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid session")
    if request.method not in _SAFE_METHODS and not request.headers.get(CSRF_HEADER):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"{CSRF_HEADER} header required")
    principal = principal_for_user(session, user_id)
    if principal is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "no active membership")
    return principal


def get_principal(
    request: Request, settings: Settings = Depends(get_settings), session: Session = Depends(get_session)
) -> Principal:
    bearer = _bearer(request)
    if bearer is None:
        from_cookie = _from_cookie(request, settings, session)
        if from_cookie is not None:
            return from_cookie
    if settings.auth_mode == "none":
        return Principal(
            subject="anonymous", role="RegulationViewer", scopes=ROLE_SCOPES["RegulationViewer"], anonymous=True
        )
    if settings.auth_mode == "api_key":
        return _from_api_key(bearer, settings)
    from safety_assistant.api.dependencies.oidc import principal_from_jwt  # lazy: optional dependency
    from safety_assistant.identity.service import user_by_subject

    principal = principal_from_jwt(bearer, settings)
    user = user_by_subject(session, principal.subject)
    if user is not None:
        return principal_for_user(session, user.id) or principal
    return principal


def require_user(principal: Principal = Depends(get_principal)) -> Principal:
    """Routes that own per-user data (conversations, uploads) need a persisted identity."""
    if principal.user_id is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "a signed-in user identity is required")
    return principal


def require_scope(scope: str):  # type: ignore[no-untyped-def]
    def _check(principal: Principal = Depends(get_principal)) -> Principal:
        if scope not in principal.scopes:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"scope '{scope}' required")
        return principal

    return _check
