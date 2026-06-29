"""Confidential user upload API (Phase A — structured test report)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import UserUpload
from registry.auth import AuthenticatedUser, get_current_user
from registry.clause_search import search_resolvable_clauses, validate_linked_clause
from registry.user_upload import create_upload_row, delete_user_upload, process_structured_upload

router = APIRouter(tags=["user-uploads"])


class UploadStatusResponse(BaseModel):
    upload_id: str
    status: str
    upload_type: str
    test_id: str | None = None
    linked_regulation_clause: str | None = None
    error_message: str | None = None
    confidential_tier: bool = True


class ClauseCandidate(BaseModel):
    regulation_code: str
    section: str
    document_name: str
    linked_regulation_clause: str
    snippet: str


@router.get("/clauses/search", response_model=list[ClauseCandidate])
def clauses_search(
    q: str,
    regulation_code: str | None = None,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    _ = current_user
    return search_resolvable_clauses(db, q=q, regulation_code=regulation_code, limit=min(limit, 50))


@router.post("/user-uploads", response_model=UploadStatusResponse)
async def create_user_upload(
    file: UploadFile = File(...),
    upload_type: str = Form("structured_test_report"),
    linked_regulation_clause: str = Form(...),
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    if upload_type != "structured_test_report":
        raise HTTPException(status_code=400, detail="Only structured_test_report is supported in Phase A")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not validate_linked_clause(db, linked_regulation_clause):
        raise HTTPException(status_code=422, detail="linked_regulation_clause is not resolvable in corpus")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    row = create_upload_row(
        db,
        user_id=current_user.user_id,
        upload_type=upload_type,
        linked_regulation_clause=linked_regulation_clause,
    )
    row = process_structured_upload(db, row, pdf_bytes)
    return UploadStatusResponse(
        upload_id=row.id,
        status=row.status,
        upload_type=row.upload_type,
        test_id=row.test_id,
        linked_regulation_clause=row.linked_regulation_clause,
        error_message=row.error_message,
        confidential_tier=row.confidential_tier,
    )


@router.get("/user-uploads/{upload_id}", response_model=UploadStatusResponse)
def get_user_upload(
    upload_id: str,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    row = db.query(UserUpload).filter(UserUpload.id == upload_id).first()
    if not row or row.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Upload not found")
    return UploadStatusResponse(
        upload_id=row.id,
        status=row.status,
        upload_type=row.upload_type,
        test_id=row.test_id,
        linked_regulation_clause=row.linked_regulation_clause,
        error_message=row.error_message,
        confidential_tier=row.confidential_tier,
    )


@router.delete("/user-uploads/{upload_id}")
def remove_user_upload(
    upload_id: str,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    row = db.query(UserUpload).filter(UserUpload.id == upload_id).first()
    if not row or row.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Upload not found")
    delete_user_upload(db, row)
    return {"ok": True, "upload_id": upload_id}
