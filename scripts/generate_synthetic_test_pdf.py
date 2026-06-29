#!/usr/bin/env python3
"""Generate synthetic structured test-report PDF for Phase A dev/tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitz  # PyMuPDF


def build_pdf(out_path: Path, record: dict) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    lines = [
        "CONFIDENTIAL CRASH TEST REPORT (SYNTHETIC)",
        f"Test ID: {record['test_id']}",
        f"Date: {record['date']}",
        f"Program: {record['program']}",
        f"Impact mode: {record['impact_mode']}",
        f"Dummy: {record.get('dummy', 'Hybrid III')}",
        "",
        "Injury measurements:",
    ]
    for res in record.get("results", []):
        lines.append(f"{res['injury_criterion']}: {res['value']} mm")
    page.insert_text((72, 72), "\n".join(lines), fontsize=11)
    doc.save(str(out_path))
    doc.close()
    return out_path


def main() -> None:
    src = ROOT / "data" / "synthetic_test_report.json"
    out = ROOT / "tests" / "fixtures" / "synthetic_test_report.pdf"
    with open(src, encoding="utf-8") as fh:
        data = json.load(fh)
    record = data[0] if isinstance(data, list) else data
    path = build_pdf(out, record)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
