"""Document workspace: uploads, listings, jobs, archive, promotion (ADR-0029 §1/§5/§6).

Every read goes through `sql_for_principal` — a document the caller may not see does not exist.
Uploads never become authoritative here; only `promote` (privileged, audited) changes scope.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from safety_assistant.config import Settings
from safety_assistant.domain.documents import DOCUMENT_TYPES, UPLOAD_SCOPES, display_status
from safety_assistant.domain.regulations import VersionStatus
from safety_assistant.identity.service import record_audit
from safety_assistant.ingestion.fetch.blobstore import BlobStore, sha256_bytes
from safety_assistant.ingestion.validation.files import PDF_MAGIC
from safety_assistant.persistence.models import IngestionJob, Regulation, RegulationVersion, SourceArtifact
from safety_assistant.retrieval.authz import sql_for_principal


class UploadRejected(ValueError):
    """Deterministic refusal at the boundary (bad magic, too large, bad scope) → HTTP 400/403."""


class NotAllowed(PermissionError):
    pass


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True)
class UploadRequest:
    data: bytes
    filename: str
    title: str
    document_type: str
    scope: str
    workspace_id: uuid.UUID | None = None
    version_label: str = "v1"
    effective_from: dt.date | None = None
    notes: str | None = None


@dataclass(frozen=True)
class UploadResult:
    document: Regulation
    version: RegulationVersion
    job: IngestionJob
    duplicate: bool


def create_upload(
    session: Session, principal: Any, req: UploadRequest, *, blob_store: BlobStore, settings: Settings
) -> UploadResult:
    if req.scope not in UPLOAD_SCOPES:
        raise UploadRejected("scope must be PRIVATE_USER or WORKSPACE; authoritative scope requires promotion")
    if req.document_type not in DOCUMENT_TYPES:
        raise UploadRejected(f"document_type must be one of {DOCUMENT_TYPES}")
    if req.scope == "WORKSPACE":
        if req.workspace_id is None or req.workspace_id not in principal.workspace_ids:
            raise NotAllowed("workspace_id is required and must be one of the caller's workspaces")
    if not req.data:
        raise UploadRejected("empty file")
    if len(req.data) > settings.ingest_max_file_bytes:
        raise UploadRejected(f"file exceeds {settings.ingest_max_file_bytes} bytes")
    if not req.data.startswith(PDF_MAGIC):
        raise UploadRejected("not a PDF (magic bytes)")

    sha = sha256_bytes(req.data)
    existing = _same_owner_duplicate(session, principal, sha, req.scope, req.workspace_id)
    if existing is not None:
        doc, version = existing
        job = _latest_job(session, version.id) or enqueue(session, version.id, requested_by=principal.user_id)
        return UploadResult(doc, version, job, duplicate=True)

    artifact = session.scalar(select(SourceArtifact).where(SourceArtifact.sha256 == sha))
    if artifact is None:
        artifact = SourceArtifact(
            sha256=sha,
            storage_uri=blob_store.put(req.data, suffix=".pdf"),
            filename=_safe_filename(req.filename),
            media_type="application/pdf",
            size_bytes=len(req.data),
            source_key=f"upload:{principal.user_id}",
            retrieved_at=_now(),
        )
        session.add(artifact)
        session.flush()
    doc = Regulation(
        regulation_key=f"DOC-{uuid.uuid4().hex[:12].upper()}",
        title=req.title.strip()[:300],
        kind=req.document_type,
        authority="USER_UPLOAD",
        jurisdiction="INTERNAL",
        authority_level="REFERENCE",
        data_class="CONFIDENTIAL",
        scope=req.scope,
        organization_id=principal.organization_ids[0],
        workspace_id=req.workspace_id if req.scope == "WORKSPACE" else None,
        owner_user_id=principal.user_id,
        metadata_={"uploaded_filename": _safe_filename(req.filename), "notes": (req.notes or "")[:2000] or None},
    )
    session.add(doc)
    session.flush()
    version = RegulationVersion(
        regulation_id=doc.id,
        source_artifact_id=artifact.id,
        version_label=req.version_label.strip()[:100] or "v1",
        valid_from=req.effective_from,
        status=VersionStatus.DISCOVERED.value,
        metadata_={"source_key": artifact.source_key},
    )
    session.add(version)
    session.flush()
    job = enqueue(session, version.id, requested_by=principal.user_id)
    return UploadResult(doc, version, job, duplicate=False)


def _safe_filename(name: str) -> str:
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(ch for ch in base if ch.isalnum() or ch in "._- ")[:200] or "upload.pdf"


def _same_owner_duplicate(
    session: Session, principal: Any, sha: str, scope: str, workspace_id: uuid.UUID | None
) -> tuple[Regulation, RegulationVersion] | None:
    stmt = (
        select(Regulation, RegulationVersion)
        .join(RegulationVersion, RegulationVersion.regulation_id == Regulation.id)
        .join(SourceArtifact, SourceArtifact.id == RegulationVersion.source_artifact_id)
        .where(SourceArtifact.sha256 == sha, Regulation.scope == scope, Regulation.archived_at.is_(None))
    )
    stmt = (
        stmt.where(Regulation.workspace_id == workspace_id)
        if scope == "WORKSPACE"
        else stmt.where(Regulation.owner_user_id == principal.user_id)
    )
    row = session.execute(stmt.order_by(RegulationVersion.created_at.desc()).limit(1)).first()
    return (row[0], row[1]) if row else None


# ---------------------------------------------------------------- jobs


def enqueue(session: Session, version_id: uuid.UUID, *, requested_by: uuid.UUID | None) -> IngestionJob:
    """One live job per version (partial unique index); a concurrent enqueue returns the live one."""
    live = _live_job(session, version_id)
    if live is not None:
        return live
    # run_after on the database clock: the worker's "due" check compares against now() in SQL.
    job = IngestionJob(version_id=version_id, status="QUEUED", run_after=func.now(), requested_by_user_id=requested_by)
    session.add(job)
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        live = _live_job(session, version_id)
        assert live is not None
        return live
    return job


def _live_job(session: Session, version_id: uuid.UUID) -> IngestionJob | None:
    return session.scalar(
        select(IngestionJob).where(
            IngestionJob.version_id == version_id, IngestionJob.status.in_(("QUEUED", "RUNNING"))
        )
    )


def _latest_job(session: Session, version_id: uuid.UUID) -> IngestionJob | None:
    return session.scalar(
        select(IngestionJob).where(IngestionJob.version_id == version_id).order_by(IngestionJob.created_at.desc())
    )


def job_for(
    session: Session, principal: Any, job_id: uuid.UUID
) -> tuple[IngestionJob, RegulationVersion, Regulation] | None:
    stmt = (
        select(IngestionJob, RegulationVersion, Regulation)
        .join(RegulationVersion, RegulationVersion.id == IngestionJob.version_id)
        .join(Regulation, Regulation.id == RegulationVersion.regulation_id)
        .where(IngestionJob.id == job_id, _visible_or_archived(principal))
    )
    row = session.execute(stmt).first()
    return (row[0], row[1], row[2]) if row else None


def retry_job(session: Session, principal: Any, job: IngestionJob, version: RegulationVersion) -> IngestionJob:
    if job.status != "FAILED":
        raise NotAllowed("only FAILED jobs can be retried; quarantined documents need a new upload")
    new = enqueue(session, version.id, requested_by=principal.user_id)
    record_audit(
        session,
        action="ingestion.retry",
        resource_type="ingestion_job",
        resource_id=str(job.id),
        actor_user_id=principal.user_id,
        metadata={"new_job_id": str(new.id)},
    )
    return new


# ---------------------------------------------------------------- listings


def _visible_or_archived(principal: Any) -> Any:
    """Listings show the caller's archived documents too (retrieval never does)."""
    from sqlalchemy import or_

    own = Regulation.owner_user_id == principal.user_id if principal.user_id else None
    return or_(sql_for_principal(principal), own) if own is not None else sql_for_principal(principal)


def list_documents(
    session: Session,
    principal: Any,
    *,
    scope: str | None = None,
    status: str | None = None,
    document_type: str | None = None,
    workspace_id: uuid.UUID | None = None,
    q: str | None = None,
    include_archived: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    stmt = select(Regulation).where(_visible_or_archived(principal))
    if not include_archived:
        stmt = stmt.where(Regulation.archived_at.is_(None))
    if scope:
        stmt = stmt.where(Regulation.scope == scope)
    if document_type:
        stmt = stmt.where(Regulation.kind == document_type)
    if workspace_id:
        stmt = stmt.where(Regulation.workspace_id == workspace_id)
    if q:
        stmt = stmt.where(Regulation.title.ilike(f"%{q}%"))
    docs = list(session.scalars(stmt.order_by(Regulation.created_at.desc()).limit(limit)).all())
    out = [document_view(session, d) for d in docs]
    return [v for v in out if not status or v["status"] == status]


def get_document(session: Session, principal: Any, document_id: uuid.UUID) -> Regulation | None:
    return session.scalar(select(Regulation).where(Regulation.id == document_id, _visible_or_archived(principal)))


def latest_version(session: Session, doc: Regulation) -> RegulationVersion | None:
    return session.scalar(
        select(RegulationVersion)
        .where(RegulationVersion.regulation_id == doc.id)
        .order_by(RegulationVersion.created_at.desc())
        .limit(1)
    )


def job_view(
    job: IngestionJob | None, version: RegulationVersion | None, doc: Regulation | None = None
) -> dict[str, Any] | None:
    if job is None:
        return None
    vstatus = version.status if version else "DISCOVERED"
    return {
        "id": str(job.id),
        "document_version_id": str(job.version_id),
        "status": job.status,
        "stage": display_status(vstatus, job_status=job.status, archived=bool(doc and doc.archived_at)),
        "attempt": job.attempt,
        "max_attempts": job.max_attempts,
        "error_code": job.error_code,
        "error_public_message": job.error_public_message,
        "diagnostic_reference": str(job.error_internal_ref) if job.error_internal_ref else None,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


def document_view(session: Session, doc: Regulation) -> dict[str, Any]:
    version = latest_version(session, doc)
    job = _latest_job(session, version.id) if version else None
    vstatus = version.status if version else "DISCOVERED"
    return {
        "id": str(doc.id),
        "document_key": doc.regulation_key,
        "title": doc.title,
        "document_type": doc.kind,
        "scope": doc.scope,
        "authority_level": doc.authority_level,
        "organization_id": str(doc.organization_id),
        "workspace_id": str(doc.workspace_id) if doc.workspace_id else None,
        "owner_user_id": str(doc.owner_user_id) if doc.owner_user_id else None,
        "status": display_status(vstatus, job_status=job.status if job else None, archived=doc.archived_at is not None),
        "version": (
            {
                "id": str(version.id),
                "label": version.version_label,
                "status": version.status,
                "valid_from": version.valid_from,
                "valid_to": version.valid_to,
                "page_count": (version.extraction_report or {}).get("processed_page_count"),
                "activated_at": version.activated_at,
                "source_sha256": _sha_for(session, version),
            }
            if version
            else None
        ),
        "latest_job": job_view(job, version, doc),
        "created_at": doc.created_at,
        "archived_at": doc.archived_at,
        "notes": (doc.metadata_ or {}).get("notes"),
    }


def _sha_for(session: Session, version: RegulationVersion) -> str | None:
    art = session.get(SourceArtifact, version.source_artifact_id)
    return art.sha256 if art else None


# ---------------------------------------------------------------- privileged


def archive(session: Session, principal: Any, doc: Regulation) -> Regulation:
    if doc.scope == "AUTHORITATIVE_ORG" and "system:admin" not in principal.scopes:
        raise NotAllowed("archiving an authoritative document requires an administrator")
    if (
        doc.scope != "AUTHORITATIVE_ORG"
        and doc.owner_user_id != principal.user_id
        and "system:admin" not in principal.scopes
    ):
        raise NotAllowed("only the owner or an administrator can archive this document")
    doc.archived_at = _now()
    record_audit(
        session,
        action="document.archive",
        resource_type="document",
        resource_id=str(doc.id),
        actor_user_id=principal.user_id,
    )
    session.flush()
    return doc


def promote(session: Session, principal: Any, doc: Regulation, *, request_id: str | None = None) -> Regulation:
    """PRIVATE_USER/WORKSPACE → AUTHORITATIVE_ORG. Requires document:promote; never automatic (FR-ADMIN-01)."""
    if "document:promote" not in principal.scopes:
        raise NotAllowed("scope 'document:promote' required")
    if doc.scope == "AUTHORITATIVE_ORG":
        return doc
    version = latest_version(session, doc)
    if version is None or version.status != VersionStatus.ACTIVE.value:
        raise NotAllowed("only a READY document can be promoted")
    before = doc.scope
    doc.scope = "AUTHORITATIVE_ORG"
    doc.organization_id = principal.organization_ids[0]
    doc.authority_level = "INTERNAL_APPROVED"
    doc.metadata_ = {**(doc.metadata_ or {}), "promoted_from": before, "promoted_by": str(principal.user_id)}
    record_audit(
        session,
        action="document.promote",
        resource_type="document",
        resource_id=str(doc.id),
        actor_user_id=principal.user_id,
        organization_id=doc.organization_id,
        request_id=request_id,
        metadata={"from_scope": before},
    )
    session.flush()
    return doc
