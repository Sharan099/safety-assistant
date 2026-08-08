"""One-shot: run checklist question 5Ã— and print retrieval diffs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("QDRANT_PATH", str(ROOT / "data" / "qdrant"))
os.environ["QDRANT_URL"] = ""
os.environ.setdefault("RERANK_PROVIDER", "none")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("QDRANT_EXACT_SEARCH", "1")
os.environ.setdefault("RETRIEVAL_REWRITE_LLM", "0")

from eval.determinism_eval import assert_determinism, run_determinism_check

rep = run_determinism_check(n=5)
print(
    json.dumps(
        {
            k: rep[k]
            for k in (
                "passed",
                "chunks_identical",
                "subqueries_identical",
                "canonical_chunk_ids",
                "n",
                "question",
            )
        },
        indent=2,
    )
)
for r in rep["runs"]:
    print(f"run {r['run']}: subs={r['subqueries']!r}")
    print(f"         chunks={r['chunk_ids']}")
assert_determinism(rep)
print("DETERMINISM OK")
