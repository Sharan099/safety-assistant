"""Identity routes: /me, preferences, browser session login/logout (ADR-0029 §6, D-012)."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from safety_assistant.api.dependencies.auth import SESSION_COOKIE, Principal, require_user
from safety_assistant.config import Settings, get_settings
from safety_assistant.identity.service import issue_session, load_identity, record_audit, user_by_email
from safety_assistant.persistence import get_session

router = APIRouter(prefix="/api/v1", tags=["identity"])


class DevLoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^@\s]+@[^@\s]+$")


class PreferencesPatch(BaseModel):
    default_workspace_id: uuid.UUID | None = None
    answer_density: Literal["concise", "standard", "detailed"] | None = None
    preferred_language: str | None = Field(default=None, min_length=2, max_length=8)
    ui_theme: Literal["light", "dark", "system"] | None = None


def set_session_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,
        samesite="lax",
        secure=settings.app_env == "production",
        path="/",
    )


@router.post("/auth/dev-login")
def dev_login(
    req: DevLoginRequest,
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Password-less login as a seeded user. Refused unless DEV_LOGIN_ENABLED (never in production)."""
    if not settings.dev_login_enabled or settings.app_env == "production":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    if not settings.session_secret:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "SESSION_SECRET is not configured")
    user = user_by_email(session, req.email)
    if user is None or user.status != "ACTIVE":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown user")
    user.last_login_at = dt.datetime.now(dt.UTC)
    record_audit(
        session,
        action="auth.dev_login",
        resource_type="user",
        resource_id=str(user.id),
        actor_user_id=user.id,
        request_id=getattr(request.state, "request_id", None),
    )
    session.commit()
    set_session_cookie(response, issue_session(user.id, settings), settings)
    return {"user_id": str(user.id), "email": user.email}


@router.post("/auth/logout")
def logout(response: Response, _: Principal = Depends(require_user)) -> dict[str, bool]:
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


def _me_payload(session: Session, principal: Principal) -> dict[str, Any]:
    assert principal.user_id is not None
    ident = load_identity(session, principal.user_id)
    assert ident is not None
    p = ident.preferences
    return {
        "user": {
            "id": str(ident.user.id),
            "email": ident.user.email,
            "display_name": ident.user.display_name,
            "roles": list(ident.roles),
            "scopes": sorted(principal.scopes),
        },
        "organizations": [{"id": str(m.organization_id), "role": m.role} for m in ident.memberships],
        "workspaces": [
            {"id": str(w.id), "name": w.name, "organization_id": str(w.organization_id)} for w in ident.workspaces
        ],
        "preferences": {
            "default_workspace_id": str(p.default_workspace_id) if p.default_workspace_id else None,
            "answer_density": p.answer_density,
            "preferred_language": p.preferred_language,
            "ui_theme": p.ui_theme,
        },
    }


@router.get("/me")
def me(principal: Principal = Depends(require_user), session: Session = Depends(get_session)) -> dict[str, Any]:
    return _me_payload(session, principal)


@router.patch("/me/preferences")
def patch_preferences(
    patch: PreferencesPatch,
    principal: Principal = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    assert principal.user_id is not None
    ident = load_identity(session, principal.user_id)
    assert ident is not None
    if patch.default_workspace_id is not None and patch.default_workspace_id not in ident.workspace_ids:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not a member of that workspace")
    for k, v in patch.model_dump(exclude_none=True).items():
        setattr(ident.preferences, k, v)
    ident.preferences.updated_at = dt.datetime.now(dt.UTC)
    session.commit()
    return _me_payload(session, principal)
