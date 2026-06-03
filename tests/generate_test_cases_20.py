#!/usr/bin/env python3
"""Build a 20-question subset (15 regulation + 5 guardrail) from the 70-question set."""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path(__file__).parent / "test_cases_70.json"
OUT = Path(__file__).parent / "test_cases_20.json"

# Diverse regulation positions across UN R14, UN R16, injury, procedural.
REG_POSITIONS = [0, 1, 2, 4, 5, 20, 21, 23, 30, 34, 40, 45, 50, 54, 58]
GUARD_POSITIONS = [0, 1, 2, 6, 8]


def main() -> None:
    if not SRC.exists():
        from tests.generate_test_cases_70 import main as gen70

        gen70()
    all_cases = json.loads(SRC.read_text(encoding="utf-8"))
    reg_cases = [c for c in all_cases if c.get("category") == "regulation"]
    guard_cases = [
        c for c in all_cases if c.get("category", "").startswith("guardrail")
    ]

    subset = [reg_cases[i] for i in REG_POSITIONS if i < len(reg_cases)]
    subset.extend(guard_cases[i] for i in GUARD_POSITIONS if i < len(guard_cases))

    assert len(subset) == 20, f"Expected 20, got {len(subset)}"
    OUT.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    n_reg = sum(1 for c in subset if c.get("category") == "regulation")
    n_guard = len(subset) - n_reg
    print(f"Wrote {len(subset)} cases ({n_reg} regulation + {n_guard} guardrail) -> {OUT}")


if __name__ == "__main__":
    main()
