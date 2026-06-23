#!/usr/bin/env python3
"""Report metadata classifier precision/recall per field on labeled set."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.metadata_classifier import classify_chunk  # noqa: E402

LABELS = ROOT / "tests" / "test_cases_metadata_validation.json"


def main() -> int:
    cases = json.loads(LABELS.read_text(encoding="utf-8"))
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for case in cases:
        meta = classify_chunk(
            regulation=case["regulation"],
            pdf_name=case["pdf_name"],
            text=case["text"],
        )
        for field, expected in case.get("expected", {}).items():
            got = meta.get(field)
            if got == expected:
                stats[field]["tp"] += 1
            else:
                stats[field]["fp"] += 1
                stats[field]["fn"] += 1
                print(f"MISS {case['regulation']}.{field}: expected {expected!r}, got {got!r}")

    print("\nPer-field scores:")
    for field, s in sorted(stats.items()):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f"  {field:20} precision={prec:.2f} recall={rec:.2f} (n={tp + fn})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
