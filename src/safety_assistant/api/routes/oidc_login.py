"""Browser sign-in through the organization's identity provider (OIDC authorization code + PKCE).

    GET /auth/oidc/login     → sets a sealed short-lived flow cookie (state, nonce, verifier) and redirects
    GET /auth/oidc/callback  → validates state, exchanges the code, validates the ID token (issuer,
                               audience, signature, expiry, nonce), maps the subject to a user, issues the
                               application session cookie and redirects to the web app

Users are created on first sign-in (subject + email). Membership is *not* implied by authentication:
`OIDC_DEFAULT_ROLE` grants a role in the default organization when set; otherwise an administrator adds
the membership and the user sees "no active membership" until then.
"""

from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from safety_assistant.api.dependencies.auth import SESSION_COOKIE
from safety_assistant.api.routes.me import set_session_cookie
from safety_assistant.config import Settings, get_settings
from safety_assistant.identity.oidc import FlowState, OidcError, authorization_url, exchange_code
from safety_assistant.identity.service import create_user, issue_session, record_audit, user_by_email, user_by_subject
from safety_assistant.persistence import get_session
from safety_assistant.persistence.models import Membership, User
from safety_assistant.persistence.models.identity import DEFAULT_ORGANIZATION_ID

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth/oidc", tags=["identity"])
FLOW_COOKIE = "oidc_flow"


@lru_cache(maxsize=1)
def http_client() -> httpx.Client:
    return httpx.Client(timeout=15)


def _configured(settings: Settings) -> None:
    if not (
        settings.oidc_issuer and settings.oidc_client_id and settings.oidc_redirect_uri and settings.session_secret
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "organization sign-in is not configured")


def _safe_return(settings: Settings, requested: str | None) -> str:
    """Only same-app relative paths are honoured — never an open redirect."""
    if requested and requested.startswith("/") and not requested.startswith("//"):
        return settings.frontend_url.rstrip("/") + requested
    return settings.frontend_url.rstrip("/") + "/app/home"


@router.get("/login")
def login(
    request: Request,
    return_to: str | None = Query(default=None, max_length=200),
    settings: Settings = Depends(get_settings),
) -> Response:
    _configured(settings)
    flow = FlowState.new(_safe_return(settings, return_to))
    try:
        url = authorization_url(settings, flow, http_client())
    except (OidcError, httpx.HTTPError) as exc:
        log.warning("oidc discovery failed: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "identity provider unavailable") from exc
    resp = RedirectResponse(url, status_code=302)
    resp.set_cookie(
        FLOW_COOKIE,
        flow.seal(settings.session_secret),
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=settings.app_env == "production",
        path="/api/v1/auth/oidc",
    )
    return resp


def _upsert_user(session: Session, claims: dict[str, Any], settings: Settings) -> User:
    subject = str(claims["sub"])
    email = str(claims.get("email") or f"{subject}@oidc.local").lower()
    user = user_by_subject(session, subject) or user_by_email(session, email)
    if user is None:
        name = str(claims.get("name") or claims.get("preferred_username") or email)
        if settings.oidc_default_role:
            user = create_user(
                session, email=email, display_name=name, role=settings.oidc_default_role, external_subject=subject
            )
        else:
            user = User(email=email, display_name=name, external_subject=subject)
            session.add(user)
            session.flush()
    elif user.external_subject is None:
        user.external_subject = subject  # first OIDC sign-in of a CLI-seeded user
    if settings.oidc_default_role and not session.get(Membership, (user.id, DEFAULT_ORGANIZATION_ID)):
        session.add(
            Membership(user_id=user.id, organization_id=DEFAULT_ORGANIZATION_ID, role=settings.oidc_default_role)
        )
    user.last_login_at = dt.datetime.now(dt.UTC)
    session.flush()
    return user


@router.get("/callback")
def callback(
    request: Request,
    code: str | None = Query(default=None, max_length=4096),
    state: str | None = Query(default=None, max_length=256),
    error: str | None = Query(default=None, max_length=100),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> Response:
    _configured(settings)
    if error:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "sign-in was refused by the identity provider")
    sealed = request.cookies.get(FLOW_COOKIE)
    if not sealed or not code or not state:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "sign-in flow is missing or expired; start again")
    try:
        flow = FlowState.unseal(sealed, settings.session_secret)
        if flow.state != state:
            raise OidcError("state mismatch")
        claims = exchange_code(settings, flow, code, http_client())
    except OidcError as exc:
        log.warning("oidc callback rejected: %s", exc)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "sign-in could not be completed") from exc
    except httpx.HTTPError as exc:
        # Only the exception's class name is logged (e.g. "ConnectTimeout"), never its args or the
        # token exchange response body — the rule below fires on "token" in the message text, not
        # on what's actually logged. nosemgrep only attaches to the line directly beneath it.
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
        log.warning("oidc token exchange unreachable: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "identity provider unavailable") from exc
    user = _upsert_user(session, claims, settings)
    record_audit(
        session,
        action="auth.oidc_login",
        resource_type="user",
        resource_id=str(user.id),
        actor_user_id=user.id,
        request_id=getattr(request.state, "request_id", None),
        metadata={"issuer": settings.oidc_issuer},
    )
    session.commit()
    resp = RedirectResponse(flow.return_to, status_code=302)
    resp.delete_cookie(FLOW_COOKIE, path="/api/v1/auth/oidc")
    set_session_cookie(resp, issue_session(user.id, settings), settings)
    return resp


@router.get("/methods")
def methods(settings: Settings = Depends(get_settings)) -> dict[str, bool]:
    """What the login page may offer. Never reveals configuration values."""
    return {
        "dev_login": settings.dev_login_enabled and settings.app_env != "production",
        "oidc": bool(settings.oidc_issuer and settings.oidc_client_id and settings.oidc_redirect_uri),
        "password": settings.password_auth_enabled,
    }


__all__ = ["FLOW_COOKIE", "SESSION_COOKIE", "router"]
