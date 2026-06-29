"""Change tracker manifest — normalized text hash dedup (FR-11…FR-13)."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from sqlalchemy.orm import Session

from database.models import SourceManifest
from registry.text_normalize import content_text_hash

ChangeStatus = Literal["NEW", "DUPLICATE", "CHANGED"]


class ChangeTracker:
    def classify(
        self,
        db: Session,
        *,
        source_url: str,
        authority: str,
        regulation_id: str,
        text: str,
        file_path: str | None = None,
        version: str | None = None,
    ) -> tuple[ChangeStatus, str, SourceManifest | None]:
        text_hash = content_text_hash(text)
        row = db.query(SourceManifest).filter(SourceManifest.source_url == source_url).first()

        if row is None:
            return "NEW", text_hash, None

        if row.content_text_hash == text_hash:
            return "DUPLICATE", text_hash, row

        return "CHANGED", text_hash, row

    def upsert(
        self,
        db: Session,
        *,
        source_url: str,
        authority: str,
        regulation_id: str,
        text_hash: str,
        file_path: str,
        status: str = "active",
        version: str | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> SourceManifest:
        row = db.query(SourceManifest).filter(SourceManifest.source_url == source_url).first()
        now = dt.datetime.now(dt.UTC)
        if row is None:
            row = SourceManifest(
                source_url=source_url,
                authority=authority,
                regulation_id=regulation_id,
                content_text_hash=text_hash,
                file_path=file_path,
                status=status,
                version=version,
                etag=etag,
                last_modified=last_modified,
                fetched_at=now,
            )
            db.add(row)
        else:
            row.content_text_hash = text_hash
            row.file_path = file_path
            row.status = status
            row.version = version
            row.etag = etag
            row.last_modified = last_modified
            row.fetched_at = now
        db.commit()
        db.refresh(row)
        return row

    def mark_superseded(self, db: Session, source_url: str) -> None:
        row = db.query(SourceManifest).filter(SourceManifest.source_url == source_url).first()
        if row:
            row.status = "superseded"
            db.commit()

    def get_remote_meta(self, db: Session, source_url: str) -> dict | None:
        row = db.query(SourceManifest).filter(SourceManifest.source_url == source_url).first()
        if not row:
            return None
        return {"etag": row.etag, "last_modified": row.last_modified, "content_length": None}
