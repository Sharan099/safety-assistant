"""Version/temporal routes: which version is effective on a date, and the
section-level diff between two versions (change-impact)."""

from __future__ import annotations

import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from safety_assistant.api.dependencies import Principal, require_scope
from safety_assistant.ingestion.diff import diff_versions, find_version, versions_of
from safety_assistant.persistence import get_session

router = APIRouter(prefix="/api/v1/regulations", tags=["versions"])


@router.get("/{regulation_key}/versions")
def list_versions(
    regulation_key: str,
    principal: Principal = Depends(require_scope("regulation:read")),
    session: Session = Depends(get_session),
    as_of: datetime.date | None = Query(default=None, description="resolve the version in force on this date"),
) -> dict[str, Any]:
    versions = versions_of(session, regulation_key)
    if not versions:
        raise HTTPException(404, "regulation not found or has no verified versions")
    day = as_of or datetime.date.today()
    effective = [
        v
        for v in versions
        if (v.valid_from is None or v.valid_from <= day) and (v.valid_to is None or v.valid_to > day)
    ]
    return {
        "regulation_key": regulation_key,
        "as_of": day,
        "effective_version": effective[-1].version_label if effective else None,
        "versions": [
            {
                "id": str(v.id),
                "label": v.version_label,
                "status": v.status,
                "series": v.series,
                "published_at": v.published_at,
                "valid_from": v.valid_from,
                "valid_to": v.valid_to,
                "superseded_by_id": str(v.superseded_by_id) if v.superseded_by_id else None,
                "amendments": v.amendments,
            }
            for v in versions
        ],
    }


@router.get("/{regulation_key}/diff")
def diff(
    regulation_key: str,
    from_label: str = Query(alias="from"),
    to_label: str = Query(alias="to"),
    principal: Principal = Depends(require_scope("regulation:read")),
    session: Session = Depends(get_session),
    full: bool = False,
) -> dict[str, Any]:
    a, b = find_version(session, regulation_key, from_label), find_version(session, regulation_key, to_label)
    if a is None or b is None:
        raise HTTPException(404, "one or both version labels not found for this regulation")
    d = diff_versions(session, a, b)
    out: dict[str, Any] = d.summary()
    if full:
        out["changes"] = [
            {
                "path": c.path,
                "kind": c.kind,
                "title": c.title,
                "normative_after": c.normative_after,
                "page_after": c.page_after,
                "diff": c.diff,
            }
            for c in d.added + d.removed + d.changed
        ]
    return out
