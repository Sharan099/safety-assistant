"""Verify every registry entry's local copy: exists, size matches, SHA-256 matches.

    uv run python scripts/maintenance/verify_registry.py

Exit code 1 if any entry fails. Run after copying PDFs into knowledge/ and in
the eval workflow after restoring the corpus from the artifact bucket.
"""

from __future__ import annotations

import pathlib
import sys

from safety_assistant.ingestion.fetch import sha256_file
from safety_assistant.ingestion.sources import load_registry


def main() -> int:
    registry = load_registry(pathlib.Path("knowledge/00_registry/sources.yaml"))
    failures = 0
    for e in registry.sources:
        path = pathlib.Path(e.local_path)
        if not path.is_file():
            print(f"MISSING  {e.source_key}: {e.local_path}")
            failures += 1
            continue
        size, digest = path.stat().st_size, sha256_file(path)
        if size != e.size_bytes or digest != e.sha256:
            print(f"MISMATCH {e.source_key}: size {size}/{e.size_bytes} sha {digest[:12]}/{e.sha256[:12]}")
            failures += 1
        else:
            print(f"OK       {e.source_key}")
    print(f"{len(registry.sources) - failures}/{len(registry.sources)} verified")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
