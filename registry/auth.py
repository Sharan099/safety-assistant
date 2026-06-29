"""Session-based authentication — server-verified identity for confidential harness access."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from app.config import settings
from database.connection import get_db
from database.models import AuthSession, User

SESSION_COOKIE_NAME = "safety_session"


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    username: str


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = db.query(User).filter(User.username == username).first()
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user


def create_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=settings.SESSION_TTL_DAYS)
    db.add(AuthSession(id=token, user_id=user.user_id, expires_at=expires_at))
    db.commit()
    return token


def delete_session(db: Session, token: str) -> None:
    db.query(AuthSession).filter(AuthSession.id == token).delete()
    db.commit()


def _session_token_from_request(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE_NAME)


def resolve_user_from_token(db: Session, token: str | None) -> AuthenticatedUser | None:
    if not token:
        return None
    row = (
        db.query(AuthSession)
        .filter(AuthSession.id == token, AuthSession.expires_at > datetime.utcnow())
        .first()
    )
    if row is None:
        return None
    user = db.query(User).filter(User.user_id == row.user_id).first()
    if user is None:
        return None
    return AuthenticatedUser(user_id=user.user_id, username=user.username)


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        max_age=settings.SESSION_TTL_DAYS * 86400,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite=settings.SESSION_COOKIE_SAMESITE,
    )


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> AuthenticatedUser:
    user = resolve_user_from_token(db, _session_token_from_request(request))
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def get_optional_user(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[AuthenticatedUser]:
    return resolve_user_from_token(db, _session_token_from_request(request))
