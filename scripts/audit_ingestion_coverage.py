#!/usr/bin/env python3
"""Ingestion coverage audit — per-document chunk counts, gaps, annotation coverage."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CHUNKS_FILE, CORPUS_DIR, DATA_DIR, OUTPUT_DIR  # noqa: E402
from backend.app.core.document_registry import INDEXED_LEGAL_CORPUS, get_document_meta  # noqa: E402

OUT = OUTPUT_DIR / "ingestion_audit_report.json"
OUT_MD = OUTPUT_DIR / "ingestion_audit_report.md"


def main() -> int:
    chunks_path = CHUNKS_FILE
    if not chunks_path.is_file():
        print(f"Missing {chunks_path}")
        return 1

    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])

    by_reg: Counter[str] = Counter()
    with_context = Counter[str]()
    with_applicability = Counter[str]()
    with_clause = Counter[str]()
    pdf_on_disk = {p.name: p for p in CORPUS_DIR.rglob("*.pdf")}

    for c in chunks:
        reg = c.get("regulation") or "UNKNOWN"
        by_reg[reg] += 1
        text = c.get("text") or ""
        if "CONTEXT:" in text[:300]:
            with_context[reg] += 1
        if "APPLICABILITY:" in text[:300]:
            with_applicability[reg] += 1
        if c.get("clause_number"):
            with_clause[reg] += 1

    manifest_path = DATA_DIR / "manifest" / "corpus_manifest.json"
    manifest_regs = set()
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for doc in manifest.get("documents", []):
            if doc.get("name", "").endswith(".pdf"):
                manifest_regs.add(doc.get("regulation", ""))

    rows: list[dict] = []
    gaps: list[str] = []

    all_regs = sorted(set(by_reg) | manifest_regs | set(INDEXED_LEGAL_CORPUS))
    for reg in all_regs:
        meta = get_document_meta(reg)
        n = by_reg.get(reg, 0)
        ctx_pct = round(100 * with_context[reg] / n, 1) if n else 0.0
        app_pct = round(100 * with_applicability[reg] / n, 1) if n else 0.0
        clause_pct = round(100 * with_clause[reg] / n, 1) if n else 0.0
        row = {
            "regulation": reg,
            "display_name": meta.display_name,
            "authority_tier": meta.authority_tier,
            "impact_mode": meta.impact_mode,
            "chunk_count": n,
            "context_annotated_pct": ctx_pct,
            "applicability_annotated_pct": app_pct,
            "clause_number_pct": clause_pct,
            "indexed_legal": reg in INDEXED_LEGAL_CORPUS,
        }
        rows.append(row)

        if reg in INDEXED_LEGAL_CORPUS and n == 0:
            gaps.append(f"INDEXED legal regulation {reg} has ZERO chunks")
        if reg == "UN_R135" and n < 50:
            gaps.append(f"UN_R135 suspiciously low chunk count: {n}")
        if reg == "UN_R95" and n < 100:
            gaps.append(
                f"UN_R95 only {n} chunks — corpus PDF is a 3-page amendment excerpt; "
                "full regulation text not indexed"
            )
        if n > 0 and ctx_pct < 90:
            gaps.append(f"{reg}: only {ctx_pct}% chunks have CONTEXT annotation")

    # PDF vs markdown
    md_dir = OUTPUT_DIR / "markdown"
    md_stems = {p.stem for p in md_dir.glob("*.md")} if md_dir.is_dir() else set()

    report = {
        "total_chunks": len(chunks),
        "pdf_count_on_disk": len(pdf_on_disk),
        "documents": rows,
        "gaps": gaps,
        "un_r135": {
            "chunk_count": by_reg.get("UN_R135", 0),
            "context_pct": round(100 * with_context["UN_R135"] / max(by_reg["UN_R135"], 1), 1),
            "sample_chunk_ids": [
                c["chunk_id"] for c in chunks if c.get("regulation") == "UN_R135"
            ][:5],
        },
        "un_r95": {
            "chunk_count": by_reg.get("UN_R95", 0),
            "pdf_pages": "3 (amendment excerpt only)",
            "note": "Replace with full UN R95 regulation PDF for complete coverage",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Ingestion coverage audit",
        "",
        f"Total chunks: **{len(chunks)}**",
        f"PDFs on disk: **{len(pdf_on_disk)}**",
        "",
        "| Regulation | Chunks | CONTEXT % | APPLICABILITY % | Clause % | Tier |",
        "|------------|--------|-----------|-----------------|----------|------|",
    ]
    for r in rows:
        if r["chunk_count"] == 0 and r["regulation"] not in INDEXED_LEGAL_CORPUS:
            continue
        lines.append(
            f"| {r['display_name']} | {r['chunk_count']} | {r['context_annotated_pct']} | "
            f"{r['applicability_annotated_pct']} | {r['clause_number_pct']} | {r['authority_tier']} |"
        )
    if gaps:
        lines.extend(["", "## Gaps", ""])
        for g in gaps:
            lines.append(f"- {g}")
    lines.extend(["", f"Full JSON: `{OUT}`"])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(OUT_MD.read_text(encoding="utf-8"))
    return 1 if gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
