"""Flag R94 (or any) chunks near figures/tables with section_number mismatches.

Walks Docling emit order, finds Figure/Table/Picture/Caption elements, then looks at
the next ~3 *chunked* clause units and checks whether the chunk's
``section_number`` matches an explicit leading clause id in its own text.

Usage::

    python scripts/audit_figure_adjacent_chunks.py \\
        --docling data/docling/UN_R94.docling.json \\
        --regulation-id UN-ECE-R94 \\
        --json-out data/vlm_figure_pass/UN_R94_audit.json

Exit code 1 when any mismatch is flagged.

The ``--json-out`` file feeds ``ingestion.vlm_figure_pass`` (selective LightOnOCR
on figure/table-adjacent pages only).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docling_core.types.doc import DoclingDocument

from ingestion.chunk import (
    _clause_start,
    chunk_document,
)
from ingestion.vlm_figure_pass import collect_figure_adjacent_pages

_LEADING_CLAUSE_RE = re.compile(
    r"(?m)^\s*(?P<num>\d+(?:\.\d+)*)\.(?:\s+|$)",
)


def load_doc(path: Path) -> DoclingDocument:
    if hasattr(DoclingDocument, "load_from_json"):
        return DoclingDocument.load_from_json(str(path))
    import json

    return DoclingDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


def explicit_clause_in_text(text: str) -> str | None:
    """First explicit leading clause number in chunk text (header or body line)."""
    if not text:
        return None
    # Prefer a true clause-start line.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        started = _clause_start(line)
        if started:
            return started[0]
        m = _LEADING_CLAUSE_RE.match(line)
        if m:
            return m.group("num")
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit figure/table-adjacent chunk sections")
    p.add_argument("--docling", type=Path, required=True)
    p.add_argument("--regulation-id", default="UN-ECE-R94")
    p.add_argument("--revision", default="Rev.3")
    p.add_argument("--window", type=int, default=3, help="Chunks after a figure/table to check")
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write machine-readable page list + suspects for vlm_figure_pass",
    )
    args = p.parse_args(argv)

    doc = load_doc(args.docling)
    chunks = chunk_document(
        doc, regulation_id=args.regulation_id, revision=args.revision
    )

    # Build parallel list of whether each iterate_items element is figure/table,
    # then map chunk boundaries via section flushes — simpler approach:
    # mark chunk indices that follow a table chunk, plus scan Docling for
    # figure pages and flag clause chunks on those pages / next pages.
    figure_adjacent_pages = collect_figure_adjacent_pages(doc, neighbor=True)
    figure_pages = set(collect_figure_adjacent_pages(doc, neighbor=False))

    suspect: list[dict[str, Any]] = []
    # 1) Chunks immediately after a table chunk
    for i, ch in enumerate(chunks):
        if ch.content_type != "table":
            continue
        for j in range(i + 1, min(len(chunks), i + 1 + args.window)):
            nxt = chunks[j]
            if nxt.content_type != "clause":
                continue
            explicit = explicit_clause_in_text(nxt.text)
            if not explicit:
                continue
            assigned = (nxt.section_number or "").strip()
            # Compare bare numbers (ignore Annex/ prefix)
            assigned_bare = assigned.split("/", 1)[-1]
            if explicit != assigned_bare and not assigned_bare.startswith(explicit + "."):
                if explicit != assigned:
                    suspect.append(
                        {
                            "reason": "after_table",
                            "chunk_id": nxt.chunk_id,
                            "section_number": assigned,
                            "explicit_in_text": explicit,
                            "page_number": nxt.page_number,
                            "preview": " ".join(nxt.text.split())[:140],
                        }
                    )

    # 2) Clause chunks on a figure page (or next page) with explicit mismatch
    for ch in chunks:
        if ch.content_type != "clause":
            continue
        page = ch.page_number
        if page is None:
            continue
        if int(page) not in figure_pages and (int(page) - 1) not in figure_pages:
            continue
        explicit = explicit_clause_in_text(ch.text)
        if not explicit:
            continue
        assigned = (ch.section_number or "").strip()
        assigned_bare = assigned.split("/", 1)[-1]
        if explicit == assigned_bare or assigned_bare.startswith(explicit + "."):
            continue
        if explicit == assigned:
            continue
        # Avoid dupes
        if any(s["chunk_id"] == ch.chunk_id for s in suspect):
            continue
        suspect.append(
            {
                "reason": "figure_adjacent_page",
                "chunk_id": ch.chunk_id,
                "section_number": assigned,
                "explicit_in_text": explicit,
                "page_number": page,
                "preview": " ".join(ch.text.split())[:140],
            }
        )

    # 3) Global leading-clause vs section_number (systemic, not only figure-adjacent)
    global_mismatches: list[dict[str, Any]] = []
    for ch in chunks:
        if ch.content_type != "clause":
            continue
        explicit = explicit_clause_in_text(ch.text)
        if not explicit:
            continue
        assigned = (ch.section_number or "").strip()
        assigned_bare = assigned.split("/", 1)[-1]
        if explicit == assigned_bare or explicit.startswith(assigned_bare + "."):
            continue
        global_mismatches.append(
            {
                "chunk_id": ch.chunk_id,
                "section_number": assigned,
                "explicit_in_text": explicit,
                "page_number": ch.page_number,
                "preview": " ".join(ch.text.split())[:140],
            }
        )

    print(f"Regulation: {args.regulation_id}  chunks={len(chunks)}")
    print(f"Figure/table pages: {len(figure_pages)}")
    print(f"Figure/table-adjacent pages (incl. +1): {len(figure_adjacent_pages)}")
    print(f"Figure/table-adjacent suspects: {len(suspect)}")
    for s in suspect:
        print(
            f"  [{s['reason']}] section={s['section_number']!r} "
            f"text_clause={s['explicit_in_text']!r} page={s['page_number']} "
            f"id={s['chunk_id']}\n    {s['preview']}"
        )
    print(f"Global leading-clause mismatches: {len(global_mismatches)}")
    for s in global_mismatches[:30]:
        print(
            f"  section={s['section_number']!r} text_clause={s['explicit_in_text']!r} "
            f"page={s['page_number']} id={s['chunk_id']}\n    {s['preview']}"
        )
    if len(global_mismatches) > 30:
        print(f"  ... {len(global_mismatches) - 30} more")

    # Fuel leakage spot-check
    fuel = [
        c
        for c in chunks
        if "slight leakage" in c.text.lower() or "30 g/min" in c.text.lower().replace(" ", "")
        or "30 g/" in c.text.lower()
    ]
    print("Fuel-leakage chunks:")
    for c in fuel:
        print(f"  section={c.section_number} page={c.page_number} id={c.chunk_id}")

    if args.json_out is not None:
        suspect_pages = sorted(
            {
                int(s["page_number"])
                for s in suspect
                if s.get("page_number") is not None
            }
        )
        payload = {
            "regulation_id": args.regulation_id,
            "revision": args.revision,
            "docling": str(args.docling),
            "figure_pages": sorted(figure_pages),
            "figure_adjacent_pages": figure_adjacent_pages,
            "suspect_pages": suspect_pages,
            "suspects": suspect,
            "global_mismatch_count": len(global_mismatches),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote audit JSON → {args.json_out}")

    return 1 if suspect or global_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
