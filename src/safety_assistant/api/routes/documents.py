"""Document workspace routes (ADR-0029 §6): upload, list, detail, archive, promote, job status, retry."""

from __future__ import annotations

import datetime
import uuid
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from safety_assistant.api.dependencies.auth import Principal, require_scope, require_user
from safety_assistant.api.middleware.ratelimit import rate_limited
from safety_assistant.config import Settings, get_settings
from safety_assistant.documents import service as docs
from safety_assistant.documents.service import NotAllowed, UploadRejected, UploadRequest
from safety_assistant.domain.documents import STAGES
from safety_assistant.ingestion.fetch.blobstore import BlobStore, blob_store_from_uri
from safety_assistant.persistence import get_session

router = APIRouter(prefix="/api/v1", tags=["documents"])


@lru_cache(maxsize=1)
def _blobs() -> BlobStore:
    return blob_store_from_uri(get_settings().artifact_store_uri)


def _load(session: Session, principal: Principal, document_id: uuid.UUID) -> Any:
    doc = docs.get_document(session, principal, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return doc


@router.post(
    "/documents", status_code=201, dependencies=[Depends(rate_limited), Depends(require_scope("document:upload"))]
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form(min_length=1, max_length=300)],
    document_type: Annotated[str, Form()] = "PROJECT_DOCUMENT",
    scope: Annotated[str, Form()] = "PRIVATE_USER",
    workspace_id: Annotated[uuid.UUID | None, Form()] = None,
    version_label: Annotated[str, Form(max_length=100)] = "v1",
    effective_from: Annotated[datetime.date | None, Form()] = None,
    notes: Annotated[str | None, Form(max_length=2000)] = None,
    principal: Principal = Depends(require_user),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    # Bound the read before anything else: the size limit is enforced on the stream, not after.
    data = await file.read(settings.ingest_max_file_bytes + 1)
    req = UploadRequest(
        data=data,
        filename=file.filename or "upload.pdf",
        title=title,
        document_type=document_type,
        scope=scope,
        workspace_id=workspace_id,
        version_label=version_label,
        effective_from=effective_from,
        notes=notes,
    )
    try:
        result = await run_in_threadpool(
            docs.create_upload, session, principal, req, blob_store=_blobs(), settings=settings
        )
    except UploadRejected as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except NotAllowed as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    session.commit()
    return {
        "document_id": str(result.document.id),
        "document_version_id": str(result.version.id),
        "ingestion_job_id": str(result.job.id),
        "status": "UPLOADED" if result.job.status == "QUEUED" else result.job.status,
        "duplicate": result.duplicate,
        "document": docs.document_view(session, result.document),
    }


@router.get("/documents")
def list_documents(
    scope: str | None = None,
    status_: str | None = Query(default=None, alias="status"),
    document_type: str | None = None,
    workspace_id: uuid.UUID | None = None,
    q: str | None = Query(default=None, max_length=200),
    include_archived: bool = False,
    principal: Principal = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    items = docs.list_documents(
        session,
        principal,
        scope=scope,
        status=status_,
        document_type=document_type,
        workspace_id=workspace_id,
        q=q,
        include_archived=include_archived,
    )
    return {"items": items, "stages": list(STAGES)}


@router.get("/documents/{document_id}")
def get_document(
    document_id: uuid.UUID, principal: Principal = Depends(require_user), session: Session = Depends(get_session)
) -> dict[str, Any]:
    doc = _load(session, principal, document_id)
    view = docs.document_view(session, doc)
    version = docs.latest_version(session, doc)
    view["extraction_report"] = (
        {
            k: v
            for k, v in (version.extraction_report or {}).items()
            if k in ("status", "processed_page_count", "failed_pages")
        }
        if version
        else None
    )
    return view


@router.post("/documents/{document_id}/archive")
def archive_document(
    document_id: uuid.UUID, principal: Principal = Depends(require_user), session: Session = Depends(get_session)
) -> dict[str, Any]:
    doc = _load(session, principal, document_id)
    try:
        docs.archive(session, principal, doc)
    except NotAllowed as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    session.commit()
    return docs.document_view(session, doc)


@router.post("/documents/{document_id}/promote")
def promote_document(
    document_id: uuid.UUID,
    request: Request,
    principal: Principal = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    doc = _load(session, principal, document_id)
    try:
        docs.promote(session, principal, doc, request_id=getattr(request.state, "request_id", None))
    except NotAllowed as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    session.commit()
    return docs.document_view(session, doc)


@router.get("/ingestion-jobs/{job_id}")
def get_job(
    job_id: uuid.UUID, principal: Principal = Depends(require_user), session: Session = Depends(get_session)
) -> dict[str, Any]:
    found = docs.job_for(session, principal, job_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    job, version, doc = found
    view = docs.job_view(job, version, doc)
    assert view is not None
    view["document_id"] = str(doc.id)
    view["stages"] = list(STAGES)
    return view


@router.post("/ingestion-jobs/{job_id}/retry", status_code=201)
def retry_job(
    job_id: uuid.UUID, principal: Principal = Depends(require_user), session: Session = Depends(get_session)
) -> dict[str, Any]:
    found = docs.job_for(session, principal, job_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    job, version, doc = found
    if doc.owner_user_id != principal.user_id and "document:ingest" not in principal.scopes:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "only the owner or an ingest-privileged role can retry")
    try:
        new = docs.retry_job(session, principal, job, version)
    except NotAllowed as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    session.commit()
    view = docs.job_view(new, version, doc)
    assert view is not None
    return view
