"""Coverage gap report vs coverage_expected.yaml (FR-17)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from database.models import Regulation

DEFAULT_EXPECTED_PATH = Path(__file__).resolve().parents[1] / "coverage_expected.yaml"


def load_expected_coverage(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_EXPECTED_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _normalize_reg_id(reg_id: str) -> str:
    rid = str(reg_id).strip().upper()
    if rid.startswith("UN_"):
        return rid
    if rid.startswith("R") and rid[1:].isdigit():
        return f"UN_{rid}"
    return rid


def build_coverage_report(db: Session, path: Path | None = None) -> dict[str, Any]:
    expected = load_expected_coverage(path)
    ingested = (
        db.query(Regulation.regulation_code, Regulation.status)
        .filter(Regulation.source_type == "UNECE")
        .all()
    )
    present_active = {
        _normalize_reg_id(code)
        for code, status in ingested
        if status == "ACTIVE"
    }
    present_any = {_normalize_reg_id(code) for code, _ in ingested}

    # Fetch documents mapping to classify complete vs partial
    from database.models import Document
    doc_rows = (
        db.query(Regulation.regulation_code, Document.document_name)
        .join(Document, Regulation.id == Document.regulation_id, isouter=True)
        .all()
    )
    doc_map = {}
    for code, doc_name in doc_rows:
        norm_code = _normalize_reg_id(code)
        if norm_code not in doc_map:
            doc_map[norm_code] = []
        if doc_name:
            doc_map[norm_code].append(doc_name)

    complete_regs = set()
    partial_regs = set()

    for rid in present_any:
        docs = doc_map.get(rid, [])
        if not rid.startswith("UN_R"):
            # NCAP, FMVSS etc. are complete if they have at least one document
            if docs:
                complete_regs.add(rid)
            else:
                partial_regs.add(rid)
        else:
            # UNECE regulations: check if any document is NOT a series/amendment
            is_complete = False
            for dname in docs:
                dname_lower = dname.lower()
                # If document is UN_R94.pdf, it does not contain 'series' or 'amend'
                if "series" not in dname_lower and "amend" not in dname_lower:
                    is_complete = True
                    break
            if is_complete:
                complete_regs.add(rid)
            else:
                partial_regs.add(rid)

    authorities_report: list[dict[str, Any]] = []
    total_expected = 0
    total_present = 0
    total_missing = 0
    total_complete = 0
    total_partial = 0

    for authority, cfg in (expected.get("authorities") or {}).items():
        if authority == "EU":
            continue  # Phase C sample focuses on UNECE ingested set per user scope
        regs = cfg.get("regulations") or []
        expected_ids = [_normalize_reg_id(r["id"]) for r in regs]
        missing = [rid for rid in expected_ids if rid not in present_any]
        missing_active = [rid for rid in expected_ids if rid not in present_active]
        
        found = [rid for rid in expected_ids if rid in present_any]
        complete_found = [rid for rid in found if rid in complete_regs]
        partial_found = [rid for rid in found if rid in partial_regs]

        total_expected += len(expected_ids)
        total_present += len(found)
        total_missing += len(missing)
        total_complete += len(complete_found)
        total_partial += len(partial_found)

        authorities_report.append(
            {
                "authority": authority,
                "expected_count": len(expected_ids),
                "ingested_count": len(found),
                "complete_count": len(complete_found),
                "partial_count": len(partial_found),
                "complete": complete_found,
                "partial": partial_found,
                "missing": missing,
                "missing_active": missing_active,
                "coverage_rate": round(len(found) / len(expected_ids), 3) if expected_ids else 1.0,
                "items": regs,
            }
        )

    return {
        "meta": expected.get("meta", {}),
        "summary": {
            "expected": total_expected,
            "ingested": total_present,
            "complete_count": total_complete,
            "partial_count": total_partial,
            "missing": total_missing,
            "coverage_rate": round(total_present / total_expected, 3) if total_expected else 1.0,
            "completeness_rate": round(total_complete / total_expected, 3) if total_expected else 1.0,
        },
        "authorities": authorities_report,
    }
