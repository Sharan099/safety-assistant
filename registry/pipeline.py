"""Gated acquisition pipeline orchestrator (FR-1…FR-13)."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from sqlalchemy.orm import Session

from crawler.allowlist import DomainNotAllowedError, assert_url_allowed
from crawler.change_check import head_metadata, remote_changed
from database.connection import SessionLocal
from registry.change_tracker import ChangeTracker
from registry.ingest_log import log_stage, new_run_id, timed_stage
from registry.storage_paths import (
    canonical_path,
    promote_to_storage,
    quarantine_file,
    staging_dir,
)
from registry.validation import ValidationOutcome, validate_pdf
from scheduler.tasks import ingest_document_task


def _download_to_staging(url: str, dest: Path, *, timeout: float = 60.0, retries: int = 3) -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; AutoSafety-RAG/1.0; +https://github.com/local/registry)"
        )
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True, verify=True) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                dest.write_bytes(response.content)
                return
        except Exception as exc:
            last_err = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Download failed for {url}: {last_err}")


def process_crawled_item(
    db: Session,
    *,
    run_id: str,
    item: dict[str, Any],
    full_fetch: bool,
    enqueue_ingest: bool = True,
) -> dict[str, Any]:
    """Run validation → storage → tracker gates for one crawled item."""
    file_path = item["file_path"]
    meta = item["metadata"]
    source_url = item["source_url"]
    authority = meta.get("source_type", "INTERNAL")
    regulation_id = meta.get("regulation_code", "")

    result: dict[str, Any] = {
        "source_url": source_url,
        "file_path": file_path,
        "outcome": "skipped",
    }

    tracker = ChangeTracker()

    if full_fetch:
        try:
            assert_url_allowed(source_url, authority)
        except DomainNotAllowedError as exc:
            log_stage(
                db,
                run_id=run_id,
                stage="allowlist",
                item=source_url,
                outcome="rejected",
                reason=str(exc),
            )
            result["outcome"] = "rejected"
            result["reason"] = str(exc)
            return result
    else:
        try:
            remote = head_metadata(source_url)
            known = tracker.get_remote_meta(db, source_url)
            if not remote_changed(known, remote):
                log_stage(
                    db,
                    run_id=run_id,
                    stage="change_check",
                    item=source_url,
                    outcome="unchanged",
                    reason="HEAD/ETag unchanged",
                )
                result["outcome"] = "unchanged"
                return result
        except Exception as exc:
            log_stage(
                db,
                run_id=run_id,
                stage="change_check",
                item=source_url,
                outcome="error",
                reason=str(exc),
            )
            result["outcome"] = "error"
            result["reason"] = str(exc)
            return result

    staging_path = Path(file_path)
    if full_fetch and not staging_path.exists():
        staging_path = staging_dir() / Path(file_path).name
        _download_to_staging(source_url, staging_path)

    with timed_stage(db, run_id=run_id, stage="validate", item=str(staging_path)) as ctx:
        validation = validate_pdf(str(staging_path), regulation_id)
        if validation.outcome == ValidationOutcome.REJECTED:
            qpath = quarantine_file(staging_path, validation.reason)
            ctx["outcome"] = "rejected"
            ctx["reason"] = validation.reason
            result.update({"outcome": "rejected", "reason": validation.reason, "quarantine": str(qpath)})
            return result
        if validation.outcome == ValidationOutcome.NEEDS_OCR:
            qpath = quarantine_file(staging_path, validation.reason)
            ctx["outcome"] = "needs_ocr"
            ctx["reason"] = validation.reason
            result.update({"outcome": "needs_ocr", "quarantine": str(qpath)})
            return result

    canonical = promote_to_storage(staging_path, meta)
    result["canonical_path"] = str(canonical)

    change_status, text_hash, _ = tracker.classify(
        db,
        source_url=source_url,
        authority=authority,
        regulation_id=regulation_id,
        text=validation.extracted_text,
        file_path=str(canonical),
        version=meta.get("amendment"),
    )

    log_stage(
        db,
        run_id=run_id,
        stage="tracker",
        item=source_url,
        outcome=change_status.lower(),
        reason=f"hash={text_hash[:12]}",
    )

    if change_status == "DUPLICATE":
        result["outcome"] = "duplicate"
        return result

    remote_meta = head_metadata(source_url) if full_fetch else {}
    tracker.upsert(
        db,
        source_url=source_url,
        authority=authority,
        regulation_id=regulation_id,
        text_hash=text_hash,
        file_path=str(canonical),
        status="active",
        version=meta.get("amendment"),
        etag=remote_meta.get("etag"),
        last_modified=remote_meta.get("last_modified"),
    )

    if enqueue_ingest:
        if os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true":
            ingest_result = ingest_document_task(str(canonical), meta)
            result["ingest"] = ingest_result
        else:
            task = ingest_document_task.delay(str(canonical), meta)
            result["ingest_task_id"] = task.id

    result["outcome"] = change_status.lower()
    return result


def run_pipeline_for_items(
    items: list[dict[str, Any]],
    *,
    full_fetch: bool,
    enqueue_ingest: bool = True,
) -> dict[str, Any]:
    run_id = new_run_id()
    db = SessionLocal()
    outcomes: list[dict[str, Any]] = []
    try:
        for item in items:
            outcomes.append(
                process_crawled_item(
                    db,
                    run_id=run_id,
                    item=item,
                    full_fetch=full_fetch,
                    enqueue_ingest=enqueue_ingest,
                )
            )
    finally:
        db.close()

    summary = {
        "run_id": run_id,
        "full_fetch": full_fetch,
        "total": len(items),
        "accepted": sum(1 for o in outcomes if o.get("outcome") in {"new", "changed"}),
        "duplicate": sum(1 for o in outcomes if o.get("outcome") == "duplicate"),
        "rejected": sum(1 for o in outcomes if o.get("outcome") == "rejected"),
        "unchanged": sum(1 for o in outcomes if o.get("outcome") == "unchanged"),
        "outcomes": outcomes,
    }
    logger.info(f"Pipeline run {run_id} summary: {summary}")
    return summary


def consolidate_legacy_downloads(root: Path | None = None) -> dict[str, Any]:
    """Copy data/downloads → staging, validate, promote to storage (one-time migration)."""
    from registry.storage_paths import downloads_dir, staging_dir as stg

    root = root or Path.cwd()
    legacy = downloads_dir(root)
    copied = 0
    promoted = 0
    quarantined = 0
    for src in legacy.glob("*"):
        if not src.is_file():
            continue
        dest = stg(root) / src.name
        shutil.copy2(src, dest)
        copied += 1
        vr = validate_pdf(str(dest), "")
        if vr.outcome != ValidationOutcome.ACCEPTED:
            quarantine_file(dest, vr.reason, root)
            quarantined += 1
            continue
        # unknown metadata — park under INTERNAL until ingested with metadata
        meta = {"source_type": "INTERNAL", "regulation_code": src.stem}
        promote_to_storage(dest, meta, root)
        promoted += 1
    return {"copied": copied, "promoted": promoted, "quarantined": quarantined}
