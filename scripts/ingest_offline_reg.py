#!/usr/bin/env python3
"""Ingest validated offline PDF(s) through staging → storage → tracker → ingest."""

from __future__ import annotations

import argparse
import os
import re
import sys
import shutil
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger
from database.connection import SessionLocal
from database.models import Document
from registry.storage_paths import promote_to_storage, staging_dir, quarantine_file
from registry.validation import validate_pdf, ValidationOutcome, _regulation_id_in_text
from registry.coverage import load_expected_coverage
from registry.version_control import compute_file_hash
from registry.text_normalize import content_text_hash
from registry.staging_manifest import gather_staging_pdfs, load_staging_manifest
from registry.confidential_paths import reject_confidential_ingest_path
from scheduler.tasks import ingest_document_task
from regulation_discovery.registry.version_parser import parse_document_metadata_from_filename


def snapshot_database() -> Path | None:
    db_path = ROOT / "safety_registry.db"
    if not db_path.exists():
        logger.warning("safety_registry.db does not exist. Skipping snapshot.")
        return None
    backup_dir = ROOT / "data" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = backup_dir / f"safety_registry_batch_backup_{ts}.db"
    shutil.copy2(db_path, snapshot)
    logger.info(f"Database snapshot created at: {snapshot}")
    return snapshot


def log_harness_and_fetch(log_entries: list[dict], snapshot_path: Path | None = None) -> None:
    log_file = ROOT / "output" / "HARNESS_AND_FETCH_LOG.md"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"## Batch Ingestion Run - {timestamp}",
        "",
    ]

    if snapshot_path:
        lines.append(f"- **Database Snapshot**: `{snapshot_path.name}`")
    lines.append("- **Validation & Ingestion Results**:")

    if not log_entries:
        lines.append("  - *No files processed.*")
    else:
        for entry in log_entries:
            filename = entry["filename"]
            status = entry["status"]
            reason = entry.get("reason", "")
            reg_code = entry.get("reg_code", "UNKNOWN")
            qpath = entry.get("quarantine_path", "")

            status_str = f"**{status}**"
            if status == "ACCEPTED":
                lines.append(
                    f"  - `{filename}` -> mapped to `{reg_code}`, status={status_str}, promoted and ingested successfully."
                )
            elif status == "DUPLICATE":
                lines.append(
                    f"  - `{filename}` -> mapped to `{reg_code}`, status={status_str}, already exists in DB. Skipped."
                )
            elif status == "REJECTED":
                lines.append(
                    f"  - `{filename}` -> mapped to `{reg_code}`, status={status_str} ({reason}). Quarantined at `{qpath}`."
                )
            elif status == "UNMAPPED":
                lines.append(
                    f"  - `{filename}` -> status={status_str} ({reason}). Quarantined at `{qpath}`."
                )
            else:
                lines.append(f"  - `{filename}` -> status={status_str} ({reason}).")

    lines.append("")

    existing_content = ""
    if log_file.exists():
        existing_content = log_file.read_text(encoding="utf-8")

    new_content = "\n".join(lines)
    if existing_content:
        log_file.write_text(new_content + "\n" + existing_content, encoding="utf-8")
    else:
        log_file.write_text("# Harness and Fetch Log\n\n" + new_content, encoding="utf-8")
    logger.info(f"Wrote execution log to {log_file}")


def _lookup_expected_reg(reg_id: str, expected_regs: dict) -> tuple[str, str, dict] | None:
    """Return (reg_id, authority, reg_info) for a known regulation id like R16."""
    norm = reg_id.upper().strip()
    if norm.startswith("UN_R"):
        norm = norm.replace("UN_", "")
    for authority, cfg in expected_regs.get("authorities", {}).items():
        for reg in cfg.get("regulations", []):
            if reg["id"].upper() == norm:
                return reg["id"], authority, reg
    return None


def resolve_regulation(file_path: Path, expected_regs: dict) -> tuple[str, str, dict] | None:
    """
    Given a PDF path and the expected_regs configuration, maps the file to a regulation.
    Returns: (reg_id, authority, reg_info) or None
    """
    filename = file_path.name
    parsed = parse_document_metadata_from_filename(filename)
    filename_code = parsed.get("regulation_code") or "UNKNOWN"

    norm_code = filename_code
    if filename_code.startswith("UN_R"):
        norm_code = filename_code.replace("UN_", "")

    def clean(s: str) -> str:
        return re.sub(r"[^\w]+", "", s).upper()

    for authority, cfg in expected_regs.get("authorities", {}).items():
        for reg in cfg.get("regulations", []):
            reg_id = reg["id"]
            if clean(norm_code) == clean(reg_id) or clean(filename_code) == clean(reg_id):
                return reg_id, authority, reg

    logger.info(f"Filename mapping failed for {filename}. Falling back to text layer search...")
    try:
        from parser.pdf_parser import PDFParser

        parser = PDFParser(str(file_path))
        pages_sample = parser.parse()
        text_sample = ""
        for p in pages_sample[:3]:
            text_sample += p.get("text", "") + "\n"
    except Exception as exc:
        logger.error(f"Failed to read text layer for mapping {filename}: {exc}")
        return None

    matched_regs = []
    for authority, cfg in expected_regs.get("authorities", {}).items():
        for reg in cfg.get("regulations", []):
            reg_id = reg["id"]
            if _regulation_id_in_text(text_sample, reg_id):
                matched_regs.append((reg_id, authority, reg))

    if len(matched_regs) == 1:
        logger.info(f"Mapped {filename} to {matched_regs[0][0]} via text layer search.")
        return matched_regs[0]
    if len(matched_regs) > 1:
        logger.warning(f"Ambiguous text mapping for {filename}: multiple candidates {[m[0] for m in matched_regs]}")
        return None

    return None


def _resolve_manual_reg_code(manual_reg_code: str, expected_regs: dict) -> tuple[str, str, dict] | None:
    reg_id = manual_reg_code.replace("UN_", "").strip().upper()
    if not reg_id.startswith("R"):
        reg_id = f"R{reg_id}"
    found = _lookup_expected_reg(reg_id, expected_regs)
    if found:
        return found
    return reg_id, "UNECE", {"id": reg_id}


def process_file(
    file_path: Path,
    expected_regs: dict,
    db_session: SessionLocal,
    manual_reg_code: str | None = None,
    manual_amend: str | None = None,
    source_url: str = "",
) -> dict:
    filename = file_path.name
    logger.info(f"Processing {filename}...")

    if manual_reg_code:
        resolved = _resolve_manual_reg_code(manual_reg_code, expected_regs)
        if not resolved:
            qpath = quarantine_file(file_path, f"Unknown regulation code override: {manual_reg_code}", ROOT)
            if file_path.exists():
                file_path.unlink()
            return {
                "filename": filename,
                "status": "UNMAPPED",
                "reason": f"Unknown regulation code override: {manual_reg_code}",
                "quarantine_path": str(qpath),
            }
        reg_id, authority, reg_info = resolved
    else:
        resolved = resolve_regulation(file_path, expected_regs)
        if not resolved:
            reason = (
                "Could not map to any expected regulation in coverage_expected.yaml. "
                "Rename to UN_R<num>_... or add an entry to data/staging/manifest.yaml"
            )
            qpath = quarantine_file(file_path, reason, ROOT)
            if file_path.exists():
                file_path.unlink()
            return {
                "filename": filename,
                "status": "UNMAPPED",
                "reason": reason,
                "quarantine_path": str(qpath),
            }
        reg_id, authority, reg_info = resolved

    try:
        checksum = compute_file_hash(str(file_path))
        existing_checksum = db_session.query(Document).filter(Document.hash == checksum).first()
        if existing_checksum:
            logger.info(f"Duplicate detected via checksum for {filename}.")
            if file_path.exists():
                file_path.unlink()
            return {
                "filename": filename,
                "status": "DUPLICATE",
                "reg_code": f"UN_{reg_id}" if authority == "UNECE" else reg_id,
            }
    except Exception as exc:
        logger.error(f"Error computing hash: {exc}")

    validation = validate_pdf(str(file_path), reg_id)
    if validation.outcome == ValidationOutcome.NEEDS_OCR:
        reason = f"needs_ocr: {validation.reason}"
        qpath = quarantine_file(file_path, reason, ROOT)
        if file_path.exists():
            file_path.unlink()
        return {
            "filename": filename,
            "status": "REJECTED",
            "reason": reason,
            "reg_code": f"UN_{reg_id}" if authority == "UNECE" else reg_id,
            "quarantine_path": str(qpath),
        }
    if validation.outcome != ValidationOutcome.ACCEPTED:
        reason = validation.reason
        qpath = quarantine_file(file_path, reason, ROOT)
        if file_path.exists():
            file_path.unlink()
        return {
            "filename": filename,
            "status": "REJECTED",
            "reason": reason,
            "reg_code": f"UN_{reg_id}" if authority == "UNECE" else reg_id,
            "quarantine_path": str(qpath),
        }

    txt_hash = content_text_hash(validation.extracted_text)
    existing_texthash = db_session.query(Document).filter(Document.content_text_hash == txt_hash).first()
    if existing_texthash:
        logger.info(f"Duplicate detected via text hash for {filename}.")
        if file_path.exists():
            file_path.unlink()
        return {
            "filename": filename,
            "status": "DUPLICATE",
            "reg_code": f"UN_{reg_id}" if authority == "UNECE" else reg_id,
        }

    filename_meta = parse_document_metadata_from_filename(filename)
    series = filename_meta.get("series_number")

    if manual_amend:
        amendment = manual_amend
    elif series is not None:
        amendment = f"{series:02d} Series"
    else:
        amendment = "Base"

    reg_code = f"UN_{reg_id}" if authority == "UNECE" else reg_id

    meta = {
        "source_type": authority,
        "regulation_code": reg_code,
        "amendment": amendment,
        "title": reg_info.get("title") or f"{authority} Standard {reg_code}",
    }
    if source_url:
        meta["source_url"] = source_url

    canonical = promote_to_storage(file_path, meta, ROOT)
    logger.info(f"Promoted to {canonical}")

    result = ingest_document_task(str(canonical), meta)
    logger.info(f"Ingest result: {result}")

    if file_path.exists() and file_path.resolve() != canonical.resolve():
        file_path.unlink()

    return {
        "filename": filename,
        "status": "ACCEPTED",
        "reg_code": reg_code,
    }


def print_summary(log_entries: list[dict], files_found: int) -> int:
    """Print human-readable summary; return exit code (0 = all ok)."""
    accepted = [e for e in log_entries if e["status"] == "ACCEPTED"]
    duplicates = [e for e in log_entries if e["status"] == "DUPLICATE"]
    rejected = [e for e in log_entries if e["status"] == "REJECTED"]
    unmapped = [e for e in log_entries if e["status"] == "UNMAPPED"]
    errors = [e for e in log_entries if e["status"] == "ERROR"]

    print("\n=== Batch Ingest Summary ===")
    print(f"PDFs in staging: {files_found}")
    print(f"  ACCEPTED:   {len(accepted)}")
    print(f"  DUPLICATE:  {len(duplicates)}")
    print(f"  REJECTED:   {len(rejected)}")
    print(f"  UNMAPPED:   {len(unmapped)}")
    print(f"  ERROR:      {len(errors)}")

    if unmapped:
        print("\nFiles needing manual mapping (add to data/staging/manifest.yaml):")
        for e in unmapped:
            print(f"  - {e['filename']}: {e.get('reason', '')}")

    if rejected:
        print("\nRejected files (quarantined):")
        for e in rejected:
            print(f"  - {e['filename']}: {e.get('reason', '')}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e['filename']}: {e.get('reason', '')}")

    if files_found == 0:
        print("\nNo PDF files found in staging. Drop files into data/staging/ and re-run.")
        return 1

    if unmapped or rejected or errors:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest offline UNECE PDF(s) from staging")
    parser.add_argument(
        "pdf_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to PDF file or staging directory (default: data/staging/)",
    )
    parser.add_argument("--regulation-code", help="Override regulation code, e.g. UN_R16 or R16")
    parser.add_argument("--amendment", help="Override amendment label, e.g. '08 Series' or 'Base'")
    parser.add_argument("--manifest", type=Path, help="Staging manifest YAML/JSON (default: data/staging/manifest.yaml)")
    parser.add_argument("--source-url", default="", help="Original source URL if known")
    args = parser.parse_args()

    target = args.pdf_path
    if target is None:
        target = staging_dir(ROOT)

    if not target.exists():
        logger.error(f"Target path does not exist: {target}")
        sys.exit(1)

    try:
        reject_confidential_ingest_path(target)
        if target.is_dir():
            for child in target.rglob("*.pdf"):
                reject_confidential_ingest_path(child)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    snapshot_path = snapshot_database()
    expected_regs = load_expected_coverage()

    is_batch = target.is_dir()
    staging_root = target if is_batch else target.parent
    manifest = load_staging_manifest(args.manifest, staging_root)

    if is_batch:
        files_to_process = gather_staging_pdfs(target)
    else:
        if target.suffix.lower() != ".pdf":
            logger.error(f"Not a PDF file: {target}")
            sys.exit(1)
        files_to_process = [target]

    log_entries: list[dict] = []
    db = SessionLocal()
    try:
        for file_path in files_to_process:
            override = manifest.get(file_path.name.lower(), {})
            manual_reg = args.regulation_code or override.get("regulation_code") or None
            manual_amend = args.amendment or override.get("amendment") or None
            source_url = args.source_url or override.get("source_url") or ""

            if is_batch and args.regulation_code and len(files_to_process) > 1:
                logger.warning(
                    "--regulation-code applies to single-file mode only; "
                    f"using manifest/filename mapping for {file_path.name}"
                )
                manual_reg = override.get("regulation_code") or None

            try:
                entry = process_file(
                    file_path,
                    expected_regs,
                    db,
                    manual_reg_code=manual_reg,
                    manual_amend=manual_amend or None,
                    source_url=source_url,
                )
                log_entries.append(entry)
            except Exception as e:
                logger.exception(f"Error processing {file_path.name}: {e}")
                log_entries.append({
                    "filename": file_path.name,
                    "status": "ERROR",
                    "reason": str(e),
                })
    finally:
        db.close()

    log_harness_and_fetch(log_entries, snapshot_path)
    exit_code = print_summary(log_entries, len(files_to_process))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
