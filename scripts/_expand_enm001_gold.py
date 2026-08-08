"""Expand enm_001 expected door requirement chunk ids from the live index."""

from __future__ import annotations

import json
import re
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "golden_set.jsonl"
COLL = "regulations"

_DOOR_RE = re.compile(r"(?i)\bdoors?\b")
_REQ_RE = re.compile(r"(?i)\b(shall|must|shall\s+not)\b")
_DEF_RE = re.compile(r'(?i)^\s*\d+(?:\.\d+)*\s+"')  # glossary definitions


def main() -> None:
    client = QdrantClient(path=str(ROOT / "data" / "qdrant"))
    pts, _ = client.scroll(
        collection_name=COLL,
        scroll_filter=qm.Filter(
            must=[
                qm.FieldCondition(
                    key="regulation_id", match=qm.MatchValue(value="UN-ECE-R95")
                )
            ]
        ),
        limit=2000,
        with_payload=True,
        with_vectors=False,
    )
    door_reqs: list[tuple[str, str, str]] = []
    for pt in pts:
        p = pt.payload or {}
        text = str(p.get("text") or "")
        cid = str(p.get("chunk_id") or "")
        sec = str(p.get("section_number") or "")
        if not cid or not _DOOR_RE.search(text):
            continue
        if not _REQ_RE.search(text):
            continue
        if _DEF_RE.search(text.strip()):
            continue
        # Prefer performance / test-condition clauses over annex approval marks.
        if re.match(r"(?i)^(preamble|annex\s*[12])$", sec.strip()):
            continue
        door_reqs.append((cid, sec, text[:120].replace("\n", " ")))

    print(f"found {len(door_reqs)} door requirement chunks")
    for cid, sec, preview in sorted(door_reqs, key=lambda t: t[1]):
        print(f"  {cid}  {sec:20}  {preview}")

    rows = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    for row in rows:
        if row.get("id") != "enm_001":
            continue
        ids = [cid for cid, _, _ in door_reqs]
        # Keep stable unique order by section then id.
        ordered = []
        seen = set()
        for cid, sec, _ in sorted(door_reqs, key=lambda t: (t[1], t[0])):
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
        row["expected_chunk_ids"] = ordered
        row["expected_sections"] = sorted(
            {sec for _, sec, _ in door_reqs if sec}, key=lambda s: s
        )
        row["primary_retrieval_metric"] = "recall@5"
        print(f"enm_001 now has {len(ordered)} expected_chunk_ids")
        break

    GOLD.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"updated {GOLD}")


if __name__ == "__main__":
    main()
