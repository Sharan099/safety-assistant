"""Dump DoclingDocument elements in emit order for page-range inspection.

Usage::

    python scripts/archive/inspect_ordering.py \\
        --docling data/docling/UN_R94.docling.json \\
        --page-from 11 --page-to 15

    # Or filter by section markers found in text:
    python scripts/archive/inspect_ordering.py \\
        --docling data/docling/UN_R94.docling.json \\
        --from-marker "5.2.1.3" --to-marker "5.2.7"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docling_core.types.doc import DoclingDocument  # noqa: E402

from ingestion.chunk import (  # noqa: E402
    _bbox_list,
    _is_heading,
    _item_text,
    _page_number,
)


def _label_str(item: Any) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return type(item).__name__
    return getattr(label, "value", None) or str(label)


def _preview(text: str, n: int = 120) -> str:
    return " ".join((text or "").split())[:n]


def load_doc(path: Path) -> DoclingDocument:
    if hasattr(DoclingDocument, "load_from_json"):
        return DoclingDocument.load_from_json(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    return DoclingDocument.model_validate(data)


def iter_rows(doc: DoclingDocument) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (item, walk_level) in enumerate(doc.iterate_items()):
        text = _item_text(item)
        rows.append(
            {
                "idx": idx,
                "type": type(item).__name__,
                "label": _label_str(item),
                "is_heading": _is_heading(item),
                "walk_level": walk_level,
                "page_number": _page_number(item),
                "bounding_box": _bbox_list(item),
                "text_preview": _preview(text),
                "text": text,
            }
        )
    return rows


def find_marker_index(rows: list[dict[str, Any]], marker: str) -> int | None:
    needle = (marker or "").strip().lower()
    if not needle:
        return None
    for row in rows:
        if needle in (row["text"] or "").lower() or needle in row["text_preview"].lower():
            return int(row["idx"])
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Inspect Docling element ordering")
    p.add_argument("--docling", type=Path, required=True)
    p.add_argument("--page-from", type=int, default=None)
    p.add_argument("--page-to", type=int, default=None)
    p.add_argument("--from-marker", type=str, default="5.2.1.3")
    p.add_argument("--to-marker", type=str, default="5.2.7")
    p.add_argument("--pad", type=int, default=5, help="Extra items before/after markers")
    p.add_argument("--jsonl", type=Path, default=None, help="Optional dump path")
    args = p.parse_args(argv)

    doc = load_doc(args.docling)
    rows = iter_rows(doc)

    start = 0
    end = len(rows) - 1
    if args.page_from is not None or args.page_to is not None:
        lo = args.page_from if args.page_from is not None else 1
        hi = args.page_to if args.page_to is not None else 10_000
        idxs = [
            r["idx"]
            for r in rows
            if r["page_number"] is not None and lo <= int(r["page_number"]) <= hi
        ]
        if idxs:
            start, end = min(idxs), max(idxs)
    else:
        i0 = find_marker_index(rows, args.from_marker)
        i1 = find_marker_index(rows, args.to_marker)
        if i0 is None:
            raise SystemExit(f"from-marker not found: {args.from_marker!r}")
        if i1 is None:
            raise SystemExit(f"to-marker not found: {args.to_marker!r}")
        start = max(0, min(i0, i1) - args.pad)
        end = min(len(rows) - 1, max(i0, i1) + args.pad)

    print(
        f"# Docling ordering dump  items[{start}:{end}]  "
        f"pages={args.page_from}-{args.page_to}  "
        f"markers={args.from_marker!r}->{args.to_marker!r}"
    )
    print(
        f"{'idx':>5} {'page':>4} {'head':>4} {'lvl':>3} {'type':<22} {'label':<16} bbox  preview"
    )
    out_rows = []
    for row in rows[start : end + 1]:
        bbox = row["bounding_box"]
        bbox_s = (
            f"[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
            if len(bbox) == 4
            else "-"
        )
        print(
            f"{row['idx']:5d} "
            f"{str(row['page_number'] or '?'):>4} "
            f"{'Y' if row['is_heading'] else '.':>4} "
            f"{row['walk_level']:3d} "
            f"{row['type']:<22} "
            f"{row['label']:<16} "
            f"{bbox_s:<22} "
            f"{row['text_preview']}"
        )
        out_rows.append(row)

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as fh:
            for row in out_rows:
                slim = {k: v for k, v in row.items() if k != "text"}
                fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
        print(f"# wrote {args.jsonl}")


if __name__ == "__main__":
    main()
