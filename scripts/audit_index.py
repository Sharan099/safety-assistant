"""Read-only audit of the Qdrant ``regulations`` collection.

Usage::

    python scripts/audit_index.py
    uv run python scripts/audit_index.py

Does not modify any points — scroll + report only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from ingestion.embed_upsert import DEFAULT_COLLECTION, get_qdrant_client

_LEADING_CLAUSE_RE = re.compile(
    # Heading form "5.2.6 Title" or body form "5.2.6. The …"
    r"(?m)^\s*(?P<num>\d+(?:\.\d+)*)(?:\.(?:\s+|$)|(?=\s+[A-Za-z\"“]))",
)


def _missing_section(val: Any) -> bool:
    if val is None:
        return True
    return not str(val).strip()


def _missing_page(val: Any) -> bool:
    return val is None or val == ""


def _missing_bbox(val: Any) -> bool:
    if val is None:
        return True
    if not isinstance(val, (list, tuple)):
        return True
    return len(val) < 4


def scroll_all_payloads(client, collection: str, *, batch: int = 256) -> list[dict[str, Any]]:
    """Scroll entire collection; return payload dicts (plus point id)."""
    out: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = dict(p.payload or {})
            payload["_point_id"] = str(p.id)
            out.append(payload)
        if offset is None:
            break
    return out


def pct(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return 100.0 * num / den


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description="Read-only Qdrant regulations index audit")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument(
        "--samples-out",
        type=Path,
        default=ROOT / "audit_samples.json",
        help="Where to dump null-section samples (default: ./audit_samples.json)",
    )
    p.add_argument("--sample-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if args.collection not in existing:
        print(f"FAIL: collection {args.collection!r} not found. Have: {sorted(existing)}")
        return 1

    print(f"Scrolling collection {args.collection!r} (payloads only, read-only)…")
    payloads = scroll_all_payloads(client, args.collection)
    total = len(payloads)
    print(f"Total points: {total}\n")

    # --- per-regulation aggregates -----------------------------------------
    by_reg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pl in payloads:
        rid = str(pl.get("regulation_id") or "(missing)").strip() or "(missing)"
        by_reg[rid].append(pl)

    print("=" * 88)
    print(f"{'regulation_id':<28} {'chunks':>8}  {'revision':<16}  last_ingested")
    print("-" * 88)
    for rid in sorted(by_reg):
        rows = by_reg[rid]
        revisions = sorted({str(r.get("revision") or "").strip() or "—" for r in rows})
        rev_s = ", ".join(revisions)
        stamps = sorted(
            {
                str(r.get("ingested_at") or "").strip()
                for r in rows
                if str(r.get("ingested_at") or "").strip()
            }
        )
        if not stamps:
            last = "—"
        elif len(stamps) == 1:
            last = stamps[0]
        else:
            last = f"{stamps[0]} .. {stamps[-1]} ({len(stamps)} values)"
        print(f"{rid:<28} {len(rows):>8}  {rev_s:<16}  {last}")
    print("=" * 88)
    print()

    # --- null section_number by regulation ---------------------------------
    print("Missing section_number (% of chunks), by regulation_id")
    print("-" * 72)
    print(f"{'regulation_id':<28} {'missing':>8} {'total':>8} {'pct':>8}")
    for rid in sorted(by_reg):
        rows = by_reg[rid]
        miss = sum(1 for r in rows if _missing_section(r.get("section_number")))
        print(f"{rid:<28} {miss:>8} {len(rows):>8} {pct(miss, len(rows)):>7.1f}%")
    print()

    # --- page / bbox globally + by reg -------------------------------------
    miss_page = sum(1 for r in payloads if _missing_page(r.get("page_number")))
    miss_bbox = sum(1 for r in payloads if _missing_bbox(r.get("bounding_box")))
    miss_either = sum(
        1
        for r in payloads
        if _missing_page(r.get("page_number")) or _missing_bbox(r.get("bounding_box"))
    )
    print("Missing page_number / bounding_box (global)")
    print("-" * 72)
    print(f"  null/missing page_number:     {miss_page:>6} / {total}  ({pct(miss_page, total):.1f}%)")
    print(f"  null/missing bounding_box:    {miss_bbox:>6} / {total}  ({pct(miss_bbox, total):.1f}%)")
    print(f"  missing page OR bbox:         {miss_either:>6} / {total}  ({pct(miss_either, total):.1f}%)")
    print()
    print("Missing page_number OR bounding_box, by regulation_id")
    print("-" * 72)
    print(f"{'regulation_id':<28} {'page%':>8} {'bbox%':>8} {'either%':>8}")
    for rid in sorted(by_reg):
        rows = by_reg[rid]
        mp = sum(1 for r in rows if _missing_page(r.get("page_number")))
        mb = sum(1 for r in rows if _missing_bbox(r.get("bounding_box")))
        me = sum(
            1
            for r in rows
            if _missing_page(r.get("page_number")) or _missing_bbox(r.get("bounding_box"))
        )
        n = len(rows)
        print(
            f"{rid:<28} {pct(mp, n):>7.1f}% {pct(mb, n):>7.1f}% {pct(me, n):>7.1f}%"
        )
    print()

    # --- random samples with null section_number ---------------------------
    null_sec = [r for r in payloads if _missing_section(r.get("section_number"))]
    rng = random.Random(args.seed)
    n_take = min(args.sample_n, len(null_sec))
    samples = rng.sample(null_sec, n_take) if null_sec else []

    dump = []
    for pl in samples:
        text = pl.get("text") or pl.get("enriched_text") or ""
        dump.append(
            {
                "point_id": pl.get("_point_id"),
                "chunk_id": pl.get("chunk_id"),
                "regulation_id": pl.get("regulation_id"),
                "revision": pl.get("revision"),
                "section_number": pl.get("section_number"),
                "section_title": pl.get("section_title"),
                "section_id": pl.get("section_id"),
                "content_type": pl.get("content_type"),
                "page_number": pl.get("page_number"),
                "bounding_box": pl.get("bounding_box"),
                "heading_path": pl.get("heading_path"),
                "text_preview": " ".join(str(text).split())[:500],
            }
        )

    args.samples_out.parent.mkdir(parents=True, exist_ok=True)
    args.samples_out.write_text(
        json.dumps(
            {
                "n_null_section_number": len(null_sec),
                "n_sampled": len(dump),
                "seed": args.seed,
                "samples": dump,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"Null section_number chunks: {len(null_sec)} / {total} "
        f"({pct(len(null_sec), total):.1f}%)"
    )
    print(f"Wrote {len(dump)} samples -> {args.samples_out}")

    # --- leading clause id vs assigned section_number ----------------------
    clause_mismatches: list[dict[str, Any]] = []
    for pl in payloads:
        if str(pl.get("content_type") or "") == "table":
            continue
        text = str(pl.get("text") or pl.get("enriched_text") or "")
        m = _LEADING_CLAUSE_RE.search(text)
        if not m:
            continue
        explicit = m.group("num")
        assigned = str(pl.get("section_number") or "").strip()
        assigned_bare = assigned.split("/", 1)[-1]
        if not assigned_bare:
            continue
        # Standing check targets numeric UNECE clause ids (the fuel-leakage bug class).
        if not re.match(r"^\d+(?:\.\d+)*$", assigned_bare):
            continue
        if not re.match(r"^\d+(?:\.\d+)*$", explicit):
            continue
        # Skip measurement-like tokens (0.308) and bare integers from tables/figures.
        if "." not in explicit or explicit.startswith("0."):
            continue
        # Exact match or child clause nested under the assigned parent → OK.
        if explicit == assigned_bare or explicit.startswith(assigned_bare + "."):
            continue
        preview = " ".join(text.split())[:160]
        clause_mismatches.append(
            {
                "chunk_id": pl.get("chunk_id"),
                "regulation_id": pl.get("regulation_id"),
                "section_number": assigned,
                "explicit_in_text": explicit,
                "page_number": pl.get("page_number"),
                "preview": preview.encode("ascii", "replace").decode("ascii"),
            }
        )
    print()
    print("Leading-clause vs section_number mismatches (sibling/uncle drift)")
    print("-" * 72)
    print(f"  flagged: {len(clause_mismatches)} / {total}")
    for row in clause_mismatches[:25]:
        print(
            f"  {row['regulation_id']} section={row['section_number']!r} "
            f"text={row['explicit_in_text']!r} page={row['page_number']} "
            f"id={row['chunk_id']}"
        )
        print(f"    {row['preview']}")
    if len(clause_mismatches) > 25:
        print(f"  ... {len(clause_mismatches) - 25} more")
    if clause_mismatches:
        warn_path = args.samples_out.with_name("section_clause_mismatches.json")
        warn_path.write_text(
            json.dumps(clause_mismatches, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote mismatch report -> {warn_path}")

    # --- GATE 1 extras: clause-as-caption + figure-adjacent metadata --------
    from ingestion.caption_guard import find_clause_numbers_in_text

    clause_as_caption = 0
    figure_meta_mismatch = 0
    by_ctype: dict[str, int] = defaultdict(int)
    for pl in payloads:
        ctype = str(pl.get("content_type") or "").strip().lower() or "(none)"
        by_ctype[ctype] += 1
        text = str(pl.get("text") or pl.get("enriched_text") or "")
        title = str(pl.get("section_title") or "")
        # Indexed caption pollution: figure chunk whose body is a numbered clause
        # without a Figure/Table label (should be 0 after Stage 1 remediation).
        if ctype == "figure":
            has_fig_label = bool(
                re.search(r"(?i)\b(figure|table)\s+\d+", text)
                or re.search(r"(?i)\b(figure|table)\s+\d+", title)
            )
            clause_hits = find_clause_numbers_in_text(text.split("\n\n", 1)[0])
            if clause_hits and not has_fig_label:
                clause_as_caption += 1
            parent = str(pl.get("parent_section_id") or "").strip()
            sec = str(pl.get("section_number") or "").strip()
            # Figure-adjacent metadata mismatch: figure with neither parent nor
            # a real section_number (orphan / mis-linked).
            if not parent and (not sec or sec.startswith("page-")):
                figure_meta_mismatch += 1

    print()
    print("GATE 1 — content_type mix / clause-as-caption / figure metadata")
    print("-" * 72)
    for ct in sorted(by_ctype):
        print(f"  content_type={ct:<12} {by_ctype[ct]:>6}")
    null_sec_pct = pct(len(null_sec), total)
    cac_pct = pct(clause_as_caption, total)
    fam_pct = pct(figure_meta_mismatch, total)
    print(f"  null section_number:          {null_sec_pct:.1f}%")
    print(f"  clause-as-caption flags:      {clause_as_caption} ({cac_pct:.1f}%)")
    print(f"  figure-adjacent mismatches:   {figure_meta_mismatch} ({fam_pct:.1f}%)")
    gate1_ok = (
        null_sec_pct == 0.0
        and clause_as_caption == 0
        and figure_meta_mismatch == 0
        and not clause_mismatches
    )
    print(f"  GATE 1: {'PASS' if gate1_ok else 'FAIL'}")

    return 0 if gate1_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
