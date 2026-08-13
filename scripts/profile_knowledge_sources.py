"""Level-3 source corpus profiler — CLAUDE_CODE_LEVEL3_INSTRUCTIONS.md §5,
PRD_LEVEL3.md §13, TRD_LEVEL3.md §6. Priority: run this before anything else.

Recursively discovers every file under `Knowledge source/` (the immutable
corpus root — see docs/ADR/0010 for the naming decision vs the Level-3 docs'
`knowledge_source/` spelling), including archive members, and writes:

    data/artifacts/source_profile.json
    data/artifacts/source_profile.parquet

Filesystem-only — no database required, safe to run before
`docker compose up postgres`.

Usage:
    uv run python scripts/profile_knowledge_sources.py
"""

from __future__ import annotations

import datetime
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from packages.ingestion.profiling import profile_knowledge_sources  # noqa: E402

CORPUS_ROOT = ROOT / "Knowledge source"
OUTPUT_JSON = ROOT / "data" / "artifacts" / "source_profile.json"
OUTPUT_PARQUET = ROOT / "data" / "artifacts" / "source_profile.parquet"


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_file_type = Counter(str(r["file_type"]) for r in rows)
    by_family = Counter(str(r["source_family"]) for r in rows)
    by_status = Counter(str(r["status"]) for r in rows)
    archive_members = sum(1 for r in rows if r["archive_member"])
    top_level = len(rows) - archive_members
    return {
        "total_rows": len(rows),
        "top_level_files": top_level,
        "archive_members": archive_members,
        "by_file_type": dict(sorted(by_file_type.items())),
        "by_source_family": dict(sorted(by_family.items())),
        "by_status": dict(sorted(by_status.items())),
    }


def main() -> None:
    if not CORPUS_ROOT.is_dir():
        raise SystemExit(f"corpus root not found: {CORPUS_ROOT}")

    rows = profile_knowledge_sources(CORPUS_ROOT)
    row_dicts = [r.to_dict() for r in rows]
    summary = _summarize(row_dicts)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "corpus_root": CORPUS_ROOT.as_posix(),
        "summary": summary,
        "rows": row_dicts,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Parquet doesn't take nested dict/list columns cleanly — JSON-encode
    # the few structured fields, keep everything else as plain scalar
    # columns. The full structured data stays in source_profile.json.
    flat_rows = []
    for r in row_dicts:
        flat = dict(r)
        for key in ("keyword_counts", "root_counts", "model_hints", "safety_issues"):
            flat[key] = json.dumps(flat[key]) if flat[key] is not None else None
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_parquet(OUTPUT_PARQUET, index=False)

    print(
        f"Profiled {summary['total_rows']} file(s) "
        f"({summary['top_level_files']} top-level, {summary['archive_members']} archive members)."
    )
    print(f"By file type: {summary['by_file_type']}")
    print(f"By source family: {summary['by_source_family']}")
    print(f"By status: {summary['by_status']}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
