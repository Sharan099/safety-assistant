"""Login / logout / session identity endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.connection import get_db
from registry.auth import (
    AuthenticatedUser,
    authenticate_user,
    clear_session_cookie,
    create_session,
    delete_session,
    get_current_user,
    set_session_cookie,
    _session_token_from_request,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)


class UserResponse(BaseModel):
    user_id: str
    username: str


@router.post("/login", response_model=UserResponse)
def login(
    body: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    user = authenticate_user(db, body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_session(db, user)
    set_session_cookie(response, token)
    return UserResponse(user_id=user.user_id, username=user.username)


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    token = _session_token_from_request(request)
    if token:
        delete_session(db, token)
    clear_session_cookie(response)
    return {"ok": True}


@router.get("/me", response_model=UserResponse)
def me(current_user: AuthenticatedUser = Depends(get_current_user)):
    return UserResponse(user_id=current_user.user_id, username=current_user.username)
