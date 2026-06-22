"""Load and normalize golden set items."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN = ROOT / "tests" / "golden_set.json"

# Default RAGAS judge subset (highest-value regression items)
DEFAULT_RAGAS_SUBSET_IDS = [
    "Q01_lookup_r94_chest",
    "Q03_mapping_r95_dummy_criteria",
    "Q04_comparison_r94_vs_fmvss208",
    "Q06_lookup_r14_anchorage_load",
    "Q13_abstain_internal_target",
]

ABSTENTION_IDS = {
    "Q13_abstain_internal_target",
    "Q14_abstain_root_cause",
    "Q15_out_of_scope",
}


def load_golden(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_GOLDEN
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"items": data, "_meta": {}}
    return data


def load_items(path: Path | None = None) -> list[dict[str, Any]]:
    return load_golden(path)["items"]


def item_by_id(items: list[dict], item_id: str) -> dict | None:
    for it in items:
        if it.get("id") == item_id:
            return it
    return None
