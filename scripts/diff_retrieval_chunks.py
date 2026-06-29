#!/usr/bin/env python3
"""Deterministic diff of retrieved chunks before/after filter fix (no LLM)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BEFORE = ROOT / "output" / "ragas_evaluation_final.retrieval.json"
AFTER = ROOT / "output" / "ragas_evaluation_tightened.retrieval.json"

HEADER_RE = re.compile(
    r"\[Source:\s*([^|]+)\|\s*Reg:\s*([^|]+)\|\s*Amendment:\s*([^|]+)\|\s*"
    r"Doc:\s*([^|]+)\|\s*Page:\s*([^|]+)\|\s*Section:\s*([^|]+)\|\s*Type:\s*([^\]]+)\]"
)


def parse_chunk(ctx: str) -> dict:
    m = HEADER_RE.search(ctx)
    if not m:
        return {"fingerprint": hashlib.md5(ctx[:200].encode(), usedforsecurity=False).hexdigest()[:12], "raw": True}
    source_type, reg, amend, doc, page, section, ctype = [x.strip() for x in m.groups()]
    body = ctx.split("]", 1)[-1].strip() if "]" in ctx else ctx
    body_sig = hashlib.md5(body[:400].encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()[:10]
    fp = f"{reg}|{doc}|{section}|{ctype}|p{page}|{body_sig}"
    return {
        "fingerprint": fp,
        "source_type": source_type,
        "regulation_code": reg,
        "document": doc,
        "page": page,
        "section": section,
        "chunk_type": ctype,
        "body_preview": body[:120].replace("\n", " "),
    }


def classify_added(case_id: str, question: str, ch: dict) -> str:
    """relevant | scope_admin | noise | cross_ref_gain"""
    sec = (ch.get("section") or "").strip()
    body = (ch.get("body_preview") or "").lower()
    reg = (ch.get("regulation_code") or "").strip()

    # Scope / admin patterns
    scope_secs = re.match(r"^(0|1|1\.\d+|2\.[0-9]\.?)$", sec)  # top-level scope/defs
    if sec in ("1", "2", "3", "4", "General") or scope_secs:
        return "scope_admin"
    if sec.startswith("Annex") and "table" not in (ch.get("chunk_type") or "").lower():
        if case_id not in ("R14_isofix_annex6",) and "annex" not in question.lower():
            # annex headers / TOC unless query names annex
            if len(body) < 80 or "diagram" in body or "contents" in body.lower():
                return "scope_admin"
    if "footnote" in body or "distinguish numbers of the contracting" in body:
        return "scope_admin"
    if "approval mark" in body and "7.6.2" not in body and case_id.startswith("R16"):
        if sec in ("4", "5.8", "14", "3"):
            return "scope_admin"

    # Question-specific relevance
    q = question.lower()
    if case_id == "R94_chest_deflection" or "r94" in q:
        if "42 mm" in body or "thcc" in body or "thorax compression" in body or sec.startswith("5.2.1.4"):
            return "relevant"
        if sec.startswith("Annex_10") or "lower leg" in body or "hybrid iii foot" in body:
            return "noise"
        if sec in ("1", "2.11") or "airbag" in body[:60]:
            return "scope_admin"
        if "deflection" in body or "thorax" in body or sec.startswith("5.2"):
            return "relevant"
    if case_id.startswith("R16") or "r16" in q or "7.6.2" in q:
        if "7.6.2" in sec or "7.6.2" in body or "locking" in body and "retractor" in body:
            return "relevant"
        if sec.startswith("6.2.5") or "2.12.4" in sec:
            return "relevant"
        if "annex 14" in body.lower() and "7.6.2" in body:
            return "relevant"
        if sec.startswith("Annex_17") or "|" in body[:30]:  # stray table fragment
            return "noise"
    if case_id == "R29_survival_space":
        if sec.startswith("5.2") or "survival space" in body or "manikin" in body:
            return "relevant"
        if sec.startswith("Annex_3") and ("impactor" in body or "roof strength" in body):
            return "noise"  # test setup not survival criteria
        if sec == "10.5":
            return "scope_admin"
    if case_id == "R14_isofix_annex6":
        if "annex" in sec.lower() or "anchorage" in body or "vehicle category" in body:
            return "relevant"
    if case_id == "R94_tcfc_lower_leg":
        if "8 kn" in body or "tcfc" in body or sec.startswith("5.2.1.8"):
            return "relevant"
    if case_id == "R44_vs_R129_child_restraints":
        if reg in ("UN_R44", "UN_R129") and ("child" in body or "restraint" in body or "i-size" in body):
            return "relevant"
    if case_id == "FMVSS208_vs_R94_chest":
        if "fmvss" in reg.lower() or "r94" in reg.lower():
            if "chest" in body or "deflection" in body or "thorax" in body or "hic" in body:
                return "relevant"
    if case_id == "R32_ocr_base_content":
        if "rear" in body or "passenger compartment" in body or "r32" in reg.lower():
            return "relevant"
    if case_id == "honesty_nonexistent_reg":
        return "scope_admin"

    if ch.get("chunk_type") == "table" and "annex" in q:
        return "relevant"
    return "unclear"


def resolve_chunk_id(ch: dict) -> str | None:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(f"sqlite:///{ROOT / 'safety_registry.db'}")
        doc = ch.get("document")
        sec = ch.get("section")
        if not doc or not sec:
            return None
        page = ch.get("page")
        with engine.connect() as c:
            row = c.execute(
                text(
                    "SELECT c.id FROM chunks c "
                    "JOIN documents d ON c.document_id=d.id "
                    "WHERE d.document_name=:doc AND c.section=:sec "
                    "AND (c.page_number=:page OR :page IS NULL OR :page='None') "
                    "LIMIT 3"
                ),
                {"doc": doc, "sec": sec, "page": None if page in (None, "None", "") else int(page)},
            ).fetchall()
            if len(row) == 1:
                return str(row[0][0])
            if row:
                return "|".join(str(r[0]) for r in row)
    except Exception:
        pass
    return None


def diff_case(case_before: dict, case_after: dict) -> dict:
    before_map = {parse_chunk(c)["fingerprint"]: parse_chunk(c) for c in case_before.get("contexts", [])}
    after_map = {parse_chunk(c)["fingerprint"]: parse_chunk(c) for c in case_after.get("contexts", [])}
    added_fps = set(after_map) - set(before_map)
    removed_fps = set(before_map) - set(after_map)
    same_fps = set(before_map) & set(after_map)

    added = []
    for fp in sorted(added_fps):
        ch = after_map[fp]
        ch["chunk_id"] = resolve_chunk_id(ch)
        ch["classification"] = classify_added(case_after["id"], case_after["question"], ch)
        added.append(ch)

    removed = []
    for fp in sorted(removed_fps):
        ch = before_map[fp]
        ch["chunk_id"] = resolve_chunk_id(ch)
        removed.append(ch)

    return {
        "id": case_after["id"],
        "question": case_after["question"],
        "before_count": len(before_map),
        "after_count": len(after_map),
        "same_count": len(same_fps),
        "jaccard": len(same_fps) / len(set(before_map) | set(after_map)) if before_map or after_map else 1.0,
        "added": added,
        "removed": removed,
    }


def audit_ocr_regs(regs: list[str]) -> dict:
    from parser.pdf_parser import PDFParser
    from vectorization.structure_chunker import StructureAwareChunker
    from sqlalchemy import create_engine, text

    engine = create_engine(f"sqlite:///{ROOT / 'safety_registry.db'}")
    chunker = StructureAwareChunker()
    out = {}
    for reg in regs:
        with engine.connect() as c:
            row = c.execute(
                text(
                    "SELECT d.file_path, d.document_name FROM documents d "
                    "JOIN regulations r ON d.regulation_id=r.id "
                    "WHERE r.regulation_code=:reg AND d.document_name LIKE '%Base%' "
                    "ORDER BY d.id LIMIT 1"
                ),
                {"reg": reg},
            ).fetchone()
        if not row:
            out[reg] = {"status": "no_base_pdf"}
            continue
        path, docname = row
        if not Path(path).exists():
            out[reg] = {"status": "file_missing", "path": path}
            continue
        pages = PDFParser(str(path)).parse()
        chunks = chunker.chunk_document(
            pages, {"regulation_code": reg, "source_type": "UNECE", "amendment": "Base"}, docname
        )
        secs_762_style = sorted(
            {c.get("section") for c in chunks if c.get("section") and re.match(r"^\d+\.\d+\.\d+", str(c.get("section")))}
        )
        # clauses cited in text but wrong section tag (Annex parent with embedded clause numbers)
        mis_tagged = []
        for c in chunks:
            sec = str(c.get("section") or "")
            body = c.get("chunk_text", "").split("]", 1)[-1] if "]" in c.get("chunk_text", "") else c.get("chunk_text", "")
            if sec.startswith("Annex") and re.search(r"\n7\.\d+\.\d*\.?\s*\n", body):
                mis_tagged.append(sec)
        with engine.connect() as c:
            db_762 = c.execute(
                text(
                    "SELECT COUNT(*) FROM chunks c JOIN documents d ON c.document_id=d.id "
                    "JOIN regulations r ON d.regulation_id=r.id "
                    "WHERE r.regulation_code=:reg AND c.section LIKE '7.6.2%'"
                ),
                {"reg": reg},
            ).scalar()
        out[reg] = {
            "document": docname,
            "dry_run_clause_sections_sample": secs_762_style[:15],
            "dry_run_deep_sections_count": len(secs_762_style),
            "db_sections_762_pattern": db_762,
            "annex_blocks_with_embedded_7x_clauses": sorted(set(mis_tagged))[:5],
            "silent_gap_risk": bool(mis_tagged) and db_762 == 0,
        }
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Deterministic retrieval chunk diff (no LLM)")
    parser.add_argument("--before", type=Path, default=BEFORE)
    parser.add_argument("--after", type=Path, default=AFTER)
    parser.add_argument("--out-json", type=Path, default=ROOT / "output" / "retrieval_chunk_diff_report.json")
    parser.add_argument("--out-txt", type=Path, default=ROOT / "output" / "retrieval_chunk_diff_report.txt")
    parser.add_argument("--skip-ocr-audit", action="store_true")
    args = parser.parse_args()

    before_rows = {r["id"]: r for r in json.loads(args.before.read_text(encoding="utf-8"))}
    after_rows = {r["id"]: r for r in json.loads(args.after.read_text(encoding="utf-8"))}

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("DETERMINISTIC RETRIEVAL CHUNK DIFF (fingerprint = reg|doc|section|type|page|body_sig)")
    emit(f"BEFORE: {args.before}")
    emit(f"AFTER:  {args.after}")
    emit("=" * 72)

    total_added_relevant = total_added_noise = total_added_scope = 0
    all_same_ratio = []

    for qid in before_rows:
        d = diff_case(before_rows[qid], after_rows[qid])
        all_same_ratio.append(d["jaccard"])
        emit(f"\n## {d['id']}")
        emit(f"Q: {d['question'][:90]}...")
        emit(f"Counts: before={d['before_count']} after={d['after_count']} same={d['same_count']} jaccard={d['jaccard']:.2f}")

        if not d["removed"] and not d["added"]:
            emit("  IDENTICAL retrieval set (all fingerprints match)")
            continue

        if d["removed"]:
            emit(f"  REMOVED ({len(d['removed'])}):")
            for ch in d["removed"]:
                cid = ch.get("chunk_id") or "?"
                emit(f"    - id={cid} | {ch.get('regulation_code')} §{ch.get('section')} | {ch.get('document')} p.{ch.get('page')}")
                emit(f"      {ch.get('body_preview', '')[:100]}...")

        if d["added"]:
            emit(f"  ADDED ({len(d['added'])}):")
            for ch in d["added"]:
                cls = ch["classification"]
                if cls == "relevant":
                    total_added_relevant += 1
                elif cls == "scope_admin":
                    total_added_scope += 1
                else:
                    total_added_noise += 1
                cid = ch.get("chunk_id") or "?"
                emit(f"    - [{cls}] id={cid} | {ch.get('regulation_code')} §{ch.get('section')} | {ch.get('document')}")
                emit(f"      {ch.get('body_preview', '')[:100]}...")

    emit("\n" + "=" * 72)
    emit("AGGREGATE")
    emit(f"  Mean Jaccard similarity: {sum(all_same_ratio)/len(all_same_ratio):.2f}")
    emit(f"  Added classified: relevant={total_added_relevant} scope_admin={total_added_scope} noise/unclear={total_added_noise}")

    identical_cases = sum(1 for qid in before_rows if not (set(parse_chunk(c)['fingerprint'] for c in before_rows[qid]['contexts']) ^ set(parse_chunk(c)['fingerprint'] for c in after_rows[qid]['contexts'])))
    emit(f"  Cases with identical sets: {identical_cases}/10")

    # Recovery spot-checks vs pre-fix baseline substantive clauses
    emit("\n" + "=" * 72)
    emit("RECOVERY SPOT-CHECKS (substantive clauses expected in AFTER top-k)")
    checks = [
        ("R94_chest_deflection", "5.2.1.4", "42 mm"),
        ("R94_tcfc_lower_leg", "5.2.1.7", "8 kn"),
        ("R44_vs_R129_child_restraints", None, "child restraint"),
        ("R16_762_crossref_regression", "7.6.2", "locking"),
    ]
    recovery_hits = 0
    recovery_total = len(checks)
    for case_id, section, body_snip in checks:
        after_ctx = after_rows.get(case_id, {}).get("contexts", [])
        found = False
        for ctx in after_ctx:
            ch = parse_chunk(ctx)
            sec = (ch.get("section") or "")
            body = (ch.get("body_preview") or "").lower()
            sec_ok = section is None or sec == section or sec.startswith(section + ".")
            body_ok = body_snip.lower() in body or body_snip.lower() in ctx.lower()
            if sec_ok and body_ok:
                found = True
                break
            if section and section in sec and body_snip.lower() in ctx.lower():
                found = True
                break
        if found:
            recovery_hits += 1
        label = f"§{section}" if section else "substantive"
        emit(f"  {case_id}: {label} + '{body_snip}' -> {'PASS' if found else 'FAIL'}")

    scope_added_total = total_added_scope
    emit(f"\n  Scope/admin chunks added (aggregate): {scope_added_total}")
    recovered = recovery_hits == recovery_total and scope_added_total <= 8
    emit(f"  RECOVERY VERDICT: {'PASS — proceed to authoritative re-score' if recovered else 'FAIL — iterate search.py before re-scoring'}")

    ocr = {}
    if not args.skip_ocr_audit:
        emit("\n" + "=" * 72)
        emit("OCR SECTION-TAG AUDIT: UN_R21, UN_R32, UN_R33 (dry-run chunker vs DB)")
        emit("=" * 72)
        ocr = audit_ocr_regs(["UN_R21", "UN_R32", "UN_R33"])
        for reg, info in ocr.items():
            emit(f"\n{reg}: {json.dumps(info, indent=2)}")

    report = {
        "before": str(args.before),
        "after": str(args.after),
        "diffs": [diff_case(before_rows[qid], after_rows[qid]) for qid in before_rows],
        "recovery_checks": checks,
        "recovery_hits": recovery_hits,
        "recovery_total": recovery_total,
        "scope_admin_added": scope_added_total,
        "recovery_verdict": "pass" if recovered else "fail",
        "ocr_audit": ocr,
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit(f"\nWrote {args.out_json}")
    emit(f"Wrote {args.out_txt}")


if __name__ == "__main__":
    main()
