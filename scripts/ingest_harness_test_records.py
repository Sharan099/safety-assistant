#!/usr/bin/env python3
"""Ingest synthetic or laboratory test-engineering records into the database with normalized validation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger
from database.connection import SessionLocal
from database.models import Test, TestResult, Regulation, Document, Chunk
from registry.harness_limits import (
    derive_pass_fail,
    extract_limit_details,
    verify_criterion_matches_clause,
)


def validate_record_schema(record: dict) -> list[str]:
    errors = []
    required_test_fields = [
        "test_id",
        "program",
        "date",
        "test_type",
        "impact_mode",
    ]
    required_result_fields = [
        "channel",
        "injury_criterion",
        "value",
        "linked_regulation_clause",
    ]
    for field in required_test_fields:
        if field not in record or record[field] is None:
            errors.append(f"Missing test metadata field: {field}")

    # Check if results is nested or denormalized
    results = record.get("results")
    if results is None:
        # Check if record has denormalized fields
        for field in required_result_fields:
            if field not in record or record[field] is None:
                errors.append(f"Missing result field: {field}")
    else:
        if not isinstance(results, list):
            errors.append("results field must be a list of measurement records")
        else:
            for idx, res in enumerate(results):
                for field in required_result_fields:
                    if field not in res or res[field] is None:
                        errors.append(f"Result {idx} missing field: {field}")

    return errors


def ingest_record_batch(
    db,
    raw_data: list[dict],
    *,
    owner_user_id: str | None = None,
    quarantine_dir: Path | None = None,
) -> dict:
    """Ingest harness records using an existing DB session (API upload path)."""
    if not isinstance(raw_data, list):
        raw_data = [raw_data]

    success_count = 0
    error_count = 0
    outcomes = []
    qdir = quarantine_dir or (ROOT / "data" / "quarantine_harness")
    qdir.mkdir(parents=True, exist_ok=True)

    for idx, rec in enumerate(raw_data):
        if owner_user_id and not rec.get("owner_user_id"):
            rec["owner_user_id"] = owner_user_id

        test_id = rec.get("test_id", f"UNKNOWN_INDEX_{idx}")
        schema_errors = validate_record_schema(rec)
        if schema_errors:
            logger.error(f"Schema validation failed for {test_id}: {schema_errors}")
            error_count += 1
            outcomes.append({"test_id": test_id, "status": "FAILED", "reason": "; ".join(schema_errors)})
            q_path = qdir / f"schema_failed_{test_id}_{int(datetime.utcnow().timestamp())}.json"
            with open(q_path, "w", encoding="utf-8") as qf:
                json.dump({"record": rec, "schema_errors": schema_errors}, qf, indent=2)
            continue

        try:
            date_val = datetime.strptime(rec["date"], "%Y-%m-%d").date()
        except ValueError:
            err_msg = f"Invalid date format: {rec['date']} (must be YYYY-MM-DD)"
            logger.error(err_msg)
            error_count += 1
            outcomes.append({"test_id": test_id, "status": "FAILED", "reason": err_msg})
            continue

        results_to_process = rec["results"] if "results" in rec else [{
            "channel": rec["channel"],
            "filter_class": rec.get("filter_class"),
            "peak_value": rec.get("peak_value"),
            "injury_criterion": rec["injury_criterion"],
            "value": rec["value"],
            "unit": rec.get("unit"),
            "linked_regulation_clause": rec["linked_regulation_clause"],
        }]

        valid_results = []
        validation_refused = False
        refusal_reason = ""

        for r_idx, res in enumerate(results_to_process):
            linked_clause = res["linked_regulation_clause"]
            if not linked_clause or "#" not in linked_clause:
                validation_refused = True
                refusal_reason = f"Result {r_idx} linked clause '{linked_clause}' format must be REG_CODE#SECTION"
                break

            reg_code, sec = linked_clause.split("#", 1)
            regs = db.query(Regulation).filter(Regulation.regulation_code == reg_code).all()
            if not regs:
                validation_refused = True
                refusal_reason = f"Linked regulation '{reg_code}' not found in database."
                break

            has_missing_effective_date = any(r.effective_date is None for r in regs)
            if has_missing_effective_date:
                validation_refused = True
                refusal_reason = f"Ambiguous/missing effective_date in regulation '{reg_code}' versions."
                break

            earliest_date = min(r.effective_date for r in regs)
            if date_val < earliest_date:
                validation_refused = True
                refusal_reason = f"Provenance gap: Test date {date_val} is before the earliest regulation version in DB ({earliest_date})."
                break

            applicable_regs = [r for r in regs if r.effective_date <= date_val]
            applicable_regs.sort(key=lambda r: r.effective_date, reverse=True)
            if not applicable_regs:
                validation_refused = True
                refusal_reason = f"No applicable regulation version found for test date {date_val}."
                break
            applicable_reg = applicable_regs[0]

            chunk = db.query(Chunk).join(Document).filter(
                Document.regulation_id == applicable_reg.id,
                Chunk.section == sec,
            ).first()
            if not chunk:
                validation_refused = True
                refusal_reason = f"Linked clause section '{sec}' not found in regulation {reg_code} version {applicable_reg.amendment or 'Base'}."
                break

            crit = res["injury_criterion"]
            limit_details = extract_limit_details(crit, chunk.chunk_text)
            if limit_details is None:
                validation_refused = True
                refusal_reason = f"Could not confidently parse limit/criterion details for '{crit}' from linked clause."
                break

            limit_val, direction, parsed_unit = limit_details
            req_unit = res.get("unit")
            if req_unit and parsed_unit and req_unit.lower() != parsed_unit.lower():
                validation_refused = True
                refusal_reason = f"Unit mismatch: Record specified '{req_unit}' but clause limit is in '{parsed_unit}'."
                break

            val = res["value"]
            derived_pass_fail = derive_pass_fail(val, limit_val, direction)
            valid_results.append({
                "channel": res["channel"],
                "filter_class": res.get("filter_class"),
                "peak_value": res.get("peak_value"),
                "injury_criterion": crit,
                "value": val,
                "pass_fail": derived_pass_fail,
                "linked_regulation_clause": linked_clause,
            })

        if validation_refused:
            logger.error(f"Validation refused for {test_id}: {refusal_reason}")
            error_count += 1
            outcomes.append({"test_id": test_id, "status": "REFUSED", "reason": refusal_reason})
            q_path = qdir / f"refused_{test_id}_{int(datetime.utcnow().timestamp())}.json"
            with open(q_path, "w", encoding="utf-8") as qf:
                json.dump({"record": rec, "refusal_reason": refusal_reason}, qf, indent=2)
            continue

        existing_test = db.query(Test).filter(Test.test_id == test_id).first()
        if existing_test:
            existing_test.program = rec["program"]
            existing_test.date = date_val
            existing_test.test_type = rec["test_type"]
            existing_test.impact_mode = rec["impact_mode"]
            existing_test.dummy = rec.get("dummy")
            existing_test.setup_revision = rec.get("setup_revision")
            existing_test.signed_off_by = rec.get("signed_off_by")
            existing_test.confidential_tier = rec.get("confidential_tier", True)
            if rec.get("owner_user_id"):
                existing_test.owner_user_id = rec["owner_user_id"]
            db.query(TestResult).filter(TestResult.test_id == test_id).delete()
        else:
            db.add(Test(
                test_id=test_id,
                program=rec["program"],
                date=date_val,
                test_type=rec["test_type"],
                impact_mode=rec["impact_mode"],
                dummy=rec.get("dummy"),
                setup_revision=rec.get("setup_revision"),
                signed_off_by=rec.get("signed_off_by"),
                confidential_tier=rec.get("confidential_tier", True),
                owner_user_id=rec.get("owner_user_id"),
            ))

        for vr in valid_results:
            db.add(TestResult(
                test_id=test_id,
                channel=vr["channel"],
                filter_class=vr["filter_class"],
                peak_value=vr["peak_value"],
                injury_criterion=vr["injury_criterion"],
                value=vr["value"],
                pass_fail=vr["pass_fail"],
                linked_regulation_clause=vr["linked_regulation_clause"],
            ))

        success_count += 1
        outcomes.append({"test_id": test_id, "status": "INGESTED"})

    db.commit()
    return {
        "total": len(raw_data),
        "success": success_count,
        "errors": error_count,
        "outcomes": outcomes,
    }


def ingest_records(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON report file not found: {json_path}")

    with open(json_path, encoding="utf-8") as fh:
        raw_data = json.load(fh)

    if not isinstance(raw_data, list):
        raw_data = [raw_data]

    db = SessionLocal()
    try:
        return ingest_record_batch(db, raw_data)
    except Exception:
        db.rollback()
        logger.exception("Transaction failed during test record ingestion")
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest harness test records")
    parser.add_argument("json_path", type=Path, help="Path to synthetic test records JSON file")
    args = parser.parse_args()

    try:
        result = ingest_records(args.json_path)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
