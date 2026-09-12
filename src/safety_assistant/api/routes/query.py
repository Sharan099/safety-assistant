"""Query routes: /search (evidence only), /ask (grounded answer), /evidence/{chunk_id},
/feedback. Retrieval and generation are CPU/IO-heavy sync code → threadpool."""

from __future__ import annotations

import datetime
import uuid
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from safety_assistant.api.dependencies import Principal, require_scope
from safety_assistant.api.middleware.ratelimit import rate_limited
from safety_assistant.conversations.service import ScopeNotAuthorized, SourceScope, validate_source_scope
from safety_assistant.generation import AnswerResponse
from safety_assistant.generation.service import AnswerService
from safety_assistant.persistence import get_session
from safety_assistant.persistence.models import (
    Chunk,
    Regulation,
    RegulationVersion,
    Section,
    SourceArtifact,
    UserFeedback,
)
from safety_assistant.retrieval import RetrievalService, ScopeFilter
from safety_assistant.retrieval.authz import Authz, sql_for_principal

router = APIRouter(prefix="/api/v1", tags=["query"])

MAX_QUERY_CHARS = 2000


@lru_cache(maxsize=1)
def _retrieval() -> RetrievalService:
    return RetrievalService()


@lru_cache(maxsize=1)
def _answers() -> AnswerService:
    return AnswerService(_retrieval())


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    as_of: datetime.date | None = None
    regulation_keys: list[str] = Field(default_factory=list, max_length=10)
    include_superseded: bool = False
    k: int | None = Field(default=None, ge=1, le=20)
    # Optional selection of authorized sources (FR-CHAT-04); ignored for principals without a user.
    source_scope: SourceScope | None = None


def scope_for(
    principal: Principal,
    *,
    as_of: datetime.date | None = None,
    regulation_keys: tuple[str, ...] = (),
    include_superseded: bool = False,
    source_scope: SourceScope | None = None,
) -> ScopeFilter:
    """Authorization narrows the universe before ranking: data class + document predicate (ADR-0029 §4)."""
    authz = None
    if principal.user_id is not None:
        ss = validate_source_scope(source_scope or SourceScope(), principal)
        authz = Authz.from_principal(
            principal,
            source_scopes=tuple(ss.scopes),
            workspace_ids=tuple(ss.workspace_ids),
            document_ids=tuple(ss.document_ids),
        )
    return ScopeFilter(
        as_of=as_of,
        regulation_keys=regulation_keys,
        include_superseded=include_superseded,
        data_classes=principal.data_classes,
        authz=authz,
    )


def _scope(req: AskRequest, principal: Principal) -> ScopeFilter:
    try:
        return scope_for(
            principal,
            as_of=req.as_of,
            regulation_keys=tuple(req.regulation_keys),
            include_superseded=req.include_superseded,
            source_scope=req.source_scope,
        )
    except ScopeNotAuthorized as exc:
        raise HTTPException(403, str(exc)) from exc


@router.post("/search", dependencies=[Depends(rate_limited)])
async def search(
    req: AskRequest,
    principal: Principal = Depends(require_scope("regulation:read")),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    result = await run_in_threadpool(_retrieval().search, session, req.query, scope=_scope(req, principal), k=req.k)
    return {
        "query": req.query,
        "scope": result.as_trace()["scope"],
        "intent": result.query_scope.intent,
        "evidence": [e.model_dump(mode="json") for e in result.bundle.evidence],
        "truncated": result.bundle.truncated,
        "latency_ms": result.latency_ms,
    }


@router.post("/ask", response_model=AnswerResponse, dependencies=[Depends(rate_limited)])
async def ask(
    req: AskRequest,
    request: Request,
    principal: Principal = Depends(require_scope("chat:query")),
    session: Session = Depends(get_session),
) -> AnswerResponse:
    resp = await run_in_threadpool(
        _answers().answer,
        session,
        req.query,
        scope=_scope(req, principal),
        principal=principal.subject,
        scopes=sorted(principal.scopes),
        k=req.k,
    )
    request.state.trace_id = resp.trace_id
    return resp


@router.get("/evidence/{chunk_id}")
def evidence(
    chunk_id: uuid.UUID,
    principal: Principal = Depends(require_scope("regulation:read")),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Resolve a citation to its exact source: section, version, artifact hash/URI."""
    row = session.execute(
        select(Chunk, Section, RegulationVersion, Regulation, SourceArtifact)
        .join(Section, Section.id == Chunk.section_id)
        .join(RegulationVersion, RegulationVersion.id == Chunk.version_id)
        .join(Regulation, Regulation.id == RegulationVersion.regulation_id)
        .join(SourceArtifact, SourceArtifact.id == RegulationVersion.source_artifact_id)
        .where(Chunk.id == chunk_id, Regulation.data_class.in_(principal.data_classes), sql_for_principal(principal))
    ).first()
    if row is None:
        raise HTTPException(404, "evidence not found")
    chunk, section, version, regulation, artifact = row
    return {
        "chunk_id": str(chunk.id),
        "citation_label": chunk.citation_label,
        "content": chunk.content,
        "chunk_sha256": chunk.chunk_sha256,
        "section": {
            "path": section.path,
            "number": section.section_number,
            "title": section.title,
            "annex": section.annex,
            "normative": section.normative,
            "page_start": section.page_start,
            "page_end": section.page_end,
        },
        "regulation": {
            "key": regulation.regulation_key,
            "title": regulation.title,
            "kind": regulation.kind,
            "jurisdiction": regulation.jurisdiction,
            "authority_level": regulation.authority_level,
        },
        "version": {
            "id": str(version.id),
            "label": version.version_label,
            "status": version.status,
            "published_at": version.published_at,
            "valid_from": version.valid_from,
            "valid_to": version.valid_to,
            "parser_version": version.parser_version,
            "chunker_version": version.chunker_version,
            "index_schema_version": version.index_schema_version,
            "amendments": version.amendments,
        },
        "source": {
            "sha256": artifact.sha256,
            "storage_uri": artifact.storage_uri,
            "source_uri": artifact.source_uri,
            "filename": artifact.filename,
            "retrieved_at": artifact.retrieved_at,
        },
    }


@router.get("/regulations")
def regulations(
    principal: Principal = Depends(require_scope("regulation:read")),
    session: Session = Depends(get_session),
    kind: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    stmt = (
        select(Regulation, RegulationVersion)
        .join(RegulationVersion, RegulationVersion.regulation_id == Regulation.id)
        .where(Regulation.data_class.in_(principal.data_classes))
        .order_by(Regulation.regulation_key, RegulationVersion.valid_from)
    )
    if kind:
        stmt = stmt.where(Regulation.kind == kind)
    out: dict[str, dict[str, Any]] = {}
    for reg, ver in session.execute(stmt).all():
        entry = out.setdefault(
            reg.regulation_key,
            {
                "regulation_key": reg.regulation_key,
                "title": reg.title,
                "kind": reg.kind,
                "jurisdiction": reg.jurisdiction,
                "authority_level": reg.authority_level,
                "versions": [],
            },
        )
        entry["versions"].append(
            {
                "id": str(ver.id),
                "label": ver.version_label,
                "status": ver.status,
                "published_at": ver.published_at,
                "valid_from": ver.valid_from,
                "valid_to": ver.valid_to,
                "activated_at": ver.activated_at,
            }
        )
    return list(out.values())


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=8, max_length=64)
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=2000)


@router.post("/feedback", status_code=201)
def feedback(
    req: FeedbackRequest,
    principal: Principal = Depends(require_scope("chat:query")),
    session: Session = Depends(get_session),
) -> dict[str, str]:
    session.add(
        UserFeedback(trace_id=req.trace_id, principal=principal.subject, rating=req.rating, comment=req.comment)
    )
    session.commit()
    return {"status": "recorded"}
