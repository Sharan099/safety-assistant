#!/usr/bin/env python3
"""Copy local safety_registry.db into data/hf/ for HF Space Docker builds (Git LFS)."""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "safety_registry.db"
DST_DIR = ROOT / "data" / "hf"
DST = DST_DIR / "safety_registry.db"


def main() -> None:
    if not SRC.is_file():
        print(f"ERROR: {SRC} not found. Run ingest first: python scripts/ingest_storage.py", file=sys.stderr)
        sys.exit(1)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    size_mb = DST.stat().st_size / (1024 * 1024)

    with sqlite3.connect(DST) as conn:
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        regs = conn.execute("SELECT COUNT(*) FROM regulations").fetchone()[0]

    print(f"Exported {DST} ({size_mb:.1f} MB)")
    print(f"  regulations: {regs}, chunks: {chunks}")
    print()
    print("Next steps for HF Space deploy:")
    print("  git lfs install")
    print("  git lfs track data/hf/safety_registry.db")
    print("  git add data/hf/safety_registry.db .gitattributes")
    print("  git commit -m 'Add HF corpus bundle (LFS)'")
    print("  git push origin main")


if __name__ == "__main__":
    main()
