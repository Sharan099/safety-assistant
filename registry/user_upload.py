"""Orchestrate confidential structured test-report uploads."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from database.models import Test, TestResult, UserUpload
from registry.clause_search import validate_linked_clause
from registry.confidential_paths import confidential_upload_dir
from registry.harness_security import audit_access
from registry.structured_test_pdf_parser import parse_structured_test_pdf
from scripts.ingest_harness_test_records import ingest_record_batch


def create_upload_row(
    db: Session,
    *,
    user_id: str,
    upload_type: str,
    linked_regulation_clause: str,
) -> UserUpload:
    upload_id = str(uuid.uuid4())
    dest_dir = confidential_upload_dir(user_id, upload_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    row = UserUpload(
        id=upload_id,
        user_id=user_id,
        upload_type=upload_type,
        status="queued",
        file_path=str(dest_dir / "original.pdf"),
        confidential_tier=True,
        linked_regulation_clause=linked_regulation_clause,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def process_structured_upload(
    db: Session,
    upload: UserUpload,
    pdf_bytes: bytes,
    *,
    model_used: str = "upload_pipeline",
) -> UserUpload:
    if not validate_linked_clause(db, upload.linked_regulation_clause or ""):
        upload.status = "failed"
        upload.error_message = "linked_regulation_clause is not resolvable in corpus"
        db.commit()
        return upload

    dest = Path(upload.file_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pdf_bytes)

    audit_access(
        db,
        user_id=upload.user_id,
        resource=upload.id,
        action="UPLOAD",
        model_used=model_used,
        details=f"type={upload.upload_type}; file={dest.name}",
    )

    upload.status = "parsing"
    db.commit()

    records, parse_errors = parse_structured_test_pdf(
        dest,
        linked_regulation_clause=upload.linked_regulation_clause or "",
    )
    manifest_path = dest.parent / "parse_manifest.json"

    if parse_errors:
        upload.status = "quarantined"
        upload.error_message = "; ".join(parse_errors)
        qdir = dest.parent / "quarantine"
        qdir.mkdir(exist_ok=True)
        shutil.copy2(dest, qdir / "original.pdf")
        manifest_path.write_text(
            json.dumps({"errors": parse_errors, "records": records}, indent=2),
            encoding="utf-8",
        )
        db.commit()
        return upload

    for rec in records:
        rec["owner_user_id"] = upload.user_id
        rec["linked_regulation_clause"] = upload.linked_regulation_clause
        for r in rec.get("results", []):
            r["linked_regulation_clause"] = upload.linked_regulation_clause

    outcome = ingest_record_batch(
        db,
        records,
        owner_user_id=upload.user_id,
        quarantine_dir=dest.parent / "quarantine",
    )
    manifest_path.write_text(json.dumps(outcome, indent=2), encoding="utf-8")

    if outcome.get("success", 0) < 1:
        upload.status = "quarantined"
        reasons = [o.get("reason", "") for o in outcome.get("outcomes", []) if o.get("status") != "INGESTED"]
        upload.error_message = "; ".join(r for r in reasons if r) or "Harness ingest refused"
        db.commit()
        return upload

    ingested = next((o for o in outcome.get("outcomes", []) if o.get("status") == "INGESTED"), None)
    upload.test_id = ingested.get("test_id") if ingested else records[0].get("test_id")
    upload.status = "ready"
    upload.error_message = None
    db.commit()
    return upload


def delete_user_upload(db: Session, upload: UserUpload) -> None:
    """Retention delete: DB rows + on-disk files."""
    test_id = upload.test_id
    if test_id:
        test = db.query(Test).filter(Test.test_id == test_id, Test.owner_user_id == upload.user_id).first()
        if test:
            db.query(TestResult).filter(TestResult.test_id == test_id).delete()
            db.delete(test)

    path = Path(upload.file_path)
    if path.exists():
        shutil.rmtree(path.parent, ignore_errors=True)

    audit_access(
        db,
        user_id=upload.user_id,
        resource=upload.id,
        action="DELETE",
        model_used="upload_pipeline",
        details=f"test_id={test_id}",
    )
    db.delete(upload)
    db.commit()
