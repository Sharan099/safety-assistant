"""Users, organizations, memberships, browser sessions, audit (ADR-0029 §3–4).

Pure data access; HTTP concerns stay in api/. Every function takes the caller's Session.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Any

import jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from safety_assistant.config import Settings
from safety_assistant.persistence.models import (
    AuditEvent,
    Membership,
    Organization,
    User,
    UserPreference,
    Workspace,
    WorkspaceMembership,
)
from safety_assistant.persistence.models.identity import DEFAULT_ORGANIZATION_ID, ORG_ROLES
from safety_assistant.security.passwords import hash_password, needs_rehash, verify_password


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True)
class Identity:
    """Resolved identity for one request: what the user is and what they belong to."""

    user: User
    memberships: tuple[Membership, ...]
    workspaces: tuple[Workspace, ...]
    preferences: UserPreference

    @property
    def organization_ids(self) -> tuple[uuid.UUID, ...]:
        return tuple(m.organization_id for m in self.memberships)

    @property
    def workspace_ids(self) -> tuple[uuid.UUID, ...]:
        return tuple(w.id for w in self.workspaces)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted({m.role for m in self.memberships}))


def default_organization(session: Session) -> Organization:
    org = session.get(Organization, DEFAULT_ORGANIZATION_ID)
    if org is None:
        org = Organization(id=DEFAULT_ORGANIZATION_ID, name="default")
        session.add(org)
        session.flush()
    return org


def create_user(
    session: Session,
    *,
    email: str,
    display_name: str,
    role: str,
    organization: Organization | None = None,
    external_subject: str | None = None,
    password: str | None = None,
) -> User:
    if role not in ORG_ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ORG_ROLES}")
    org = organization or default_organization(session)
    user = User(
        email=email.lower(),
        display_name=display_name,
        external_subject=external_subject,
        password_hash=hash_password(password) if password else None,
    )
    session.add(user)
    session.flush()
    session.add(Membership(user_id=user.id, organization_id=org.id, role=role))
    session.add(UserPreference(user_id=user.id, updated_at=_now()))
    session.flush()
    return user


def user_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(User.email == email.lower()))


def user_by_subject(session: Session, subject: str) -> User | None:
    return session.scalar(select(User).where(User.external_subject == subject))


@dataclass(frozen=True)
class AuthResult:
    """Every outcome of a password sign-in attempt, so the route can log/respond without
    re-deriving why: `ok` is the only thing shown to the client (generic "invalid credentials"),
    `reason` is for the audit event."""

    ok: bool
    user: User | None = None
    reason: str = ""


_DUMMY_HASH = hash_password("not-a-real-password-just-for-timing")  # verified on unknown-email attempts


def authenticate_user(session: Session, settings: Settings, *, email: str, password: str) -> AuthResult:
    """Email/password check with per-account lockout. Always runs a scrypt verification, even for an
    unknown email or a locked account, so response time does not reveal which case applies."""
    user = user_by_email(session, email)
    now = _now()
    if user is None or user.password_hash is None:
        verify_password(password, _DUMMY_HASH)
        return AuthResult(False, reason="no_such_account")
    if user.locked_until is not None and user.locked_until > now:
        verify_password(password, user.password_hash)
        return AuthResult(False, reason="locked")
    if user.status != "ACTIVE":
        verify_password(password, user.password_hash)
        return AuthResult(False, reason="inactive")
    if not verify_password(password, user.password_hash):
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= settings.login_max_attempts:
            user.locked_until = now + dt.timedelta(minutes=settings.login_lockout_minutes)
        session.flush()
        return AuthResult(False, reason="bad_password")
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = now
    session.flush()
    return AuthResult(True, user=user)


def set_password(user: User, password: str) -> None:
    user.password_hash = hash_password(password)
    user.failed_login_attempts = 0
    user.locked_until = None


def create_workspace(session: Session, *, organization_id: uuid.UUID, name: str, owner: User) -> Workspace:
    ws = Workspace(organization_id=organization_id, name=name, created_by=owner.id)
    session.add(ws)
    session.flush()
    session.add(WorkspaceMembership(workspace_id=ws.id, user_id=owner.id, role="owner"))
    session.flush()
    return ws


def load_identity(session: Session, user_id: uuid.UUID) -> Identity | None:
    user = session.get(User, user_id)
    if user is None or user.status != "ACTIVE":
        return None
    memberships = tuple(session.scalars(select(Membership).where(Membership.user_id == user_id)).all())
    workspaces = tuple(
        session.scalars(
            select(Workspace)
            .join(WorkspaceMembership, WorkspaceMembership.workspace_id == Workspace.id)
            .where(WorkspaceMembership.user_id == user_id, Workspace.archived_at.is_(None))
        ).all()
    )
    prefs = session.get(UserPreference, user_id)
    if prefs is None:
        prefs = UserPreference(user_id=user_id, updated_at=_now())
        session.add(prefs)
        session.flush()
    return Identity(user=user, memberships=memberships, workspaces=workspaces, preferences=prefs)


# ---------------------------------------------------------------- browser sessions

_ALG = "HS256"


def issue_session(user_id: uuid.UUID, settings: Settings, *, now: dt.datetime | None = None) -> str:
    now = now or _now()
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(hours=settings.session_ttl_hours)).timestamp()),
        "typ": "session",
    }
    return jwt.encode(payload, settings.session_secret, algorithm=_ALG)


def verify_session(token: str, settings: Settings) -> uuid.UUID | None:
    """Returns the user id or None for any invalid/expired/foreign token."""
    if not settings.session_secret:
        return None
    try:
        claims = jwt.decode(token, settings.session_secret, algorithms=[_ALG], options={"require": ["exp", "sub"]})
        if claims.get("typ") != "session":
            return None
        return uuid.UUID(str(claims["sub"]))
    except (jwt.PyJWTError, ValueError):
        return None


# ---------------------------------------------------------------- audit


def record_audit(
    session: Session,
    *,
    action: str,
    resource_type: str,
    resource_id: str | None,
    actor_user_id: uuid.UUID | None = None,
    actor_subject: str | None = None,
    organization_id: uuid.UUID | None = None,
    request_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    success: bool = True,
) -> AuditEvent:
    ev = AuditEvent(
        actor_user_id=actor_user_id,
        actor_subject=actor_subject,
        organization_id=organization_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        request_id=request_id,
        metadata_=metadata,
        success=success,
    )
    session.add(ev)
    session.flush()
    return ev
