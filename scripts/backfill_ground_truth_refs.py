"""Backfill ground_truth_reference_chunks for factual_lookup + cross_regulation.

Extracts verbatim regulation text from indexed PDFs / known corpus chunks,
spot-checks that expected_answer_contains tokens appear, and writes the
field into eval/golden_set.jsonl. Does NOT call an LLM for reference answers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

from retrieval.retrieve import fetch_chunks_by_ids

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
PDFS = {
    "UN-ECE-R94": ROOT / "data" / "pdfs" / "UN_R94.pdf",
    "UN-ECE-R95": ROOT / "data" / "pdfs" / "UN_R95.pdf",
    "UN-ECE-R16": ROOT / "data" / "pdfs" / "UN_R16.pdf",
    "UN-ECE-R129": ROOT / "data" / "pdfs" / "UN_R129.pdf",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def pdf_pages(reg: str) -> list[str]:
    path = PDFS[reg]
    reader = PdfReader(str(path))
    return [p.extract_text() or "" for p in reader.pages]


def extract_around(pages: list[str], needle: str, *, window: int = 350) -> str | None:
    needle_l = needle.lower()
    for page in pages:
        idx = page.lower().find(needle_l)
        if idx < 0:
            continue
        start = max(0, idx - 40)
        end = min(len(page), idx + window)
        return _norm(page[start:end])
    return None


def extract_clause(pages: list[str], clause: str, *, max_len: int = 500) -> str | None:
    """Pull text starting at clause number like '5.2.1.4.' through next clause-ish break."""
    pat = re.compile(rf"({re.escape(clause)}\.?\s+.+?)(?=\n\s*\d+\.\d|\Z)", re.S)
    for page in pages:
        m = pat.search(page)
        if not m:
            # looser: find clause and take window
            idx = page.find(clause)
            if idx < 0:
                continue
            chunk = _norm(page[idx : idx + max_len])
            # trim at next major numbered sibling if present
            sib = re.search(rf"{re.escape(clause.rsplit('.', 1)[0])}\.\d+\.", chunk[len(clause) :])
            if sib:
                chunk = chunk[: len(clause) + sib.start()].strip()
            return chunk
        return _norm(m.group(1))[:max_len]
    return None


def chunk_text(cid: str) -> str | None:
    fetched = fetch_chunks_by_ids([cid])
    if not fetched:
        return None
    return _norm(fetched[0].text or "") or None


def spot_check(text: str, contains: list[str] | None) -> bool:
    if not contains:
        return True
    low = text.lower()
    for token in contains:
        t = str(token).strip().lower()
        if not t:
            continue
        # numeric tokens: allow thousand separators
        if re.fullmatch(r"[\d.]+", t):
            compact = re.sub(r"[,\s]", "", low)
            if t.replace(",", "") not in compact and t not in low:
                return False
        elif t not in low:
            return False
    return True


def build_refs() -> dict[str, list[str]]:
    r94 = pdf_pages("UN-ECE-R94")
    r95 = pdf_pages("UN-ECE-R95")
    r16 = pdf_pages("UN-ECE-R16")
    r129 = pdf_pages("UN-ECE-R129")

    # Prefer corpus chunk text when available; fall back to PDF clause windows.
    refs: dict[str, list[str]] = {}

    def add(case_id: str, *texts: str | None) -> None:
        cleaned = [t for t in (_norm(x) for x in texts if x) if t and len(t) > 20]
        if cleaned:
            refs[case_id] = cleaned

    # --- factual_lookup (R94 injury / definitions) ---
    add("fac_001", chunk_text("930c351518a8769b") or extract_clause(r94, "5.2.1.4"))
    add("fac_040", chunk_text("930c351518a8769b") or extract_clause(r94, "5.2.1.4"))
    add(
        "fac_002",
        extract_clause(r94, "5.2.1.7")
        or extract_around(r94, "tibia compression force criterion (TCFC)"),
    )
    add(
        "fac_041",
        extract_clause(r94, "5.2.1.7")
        or extract_around(r94, "tibia compression force criterion (TCFC)"),
    )
    add("fac_003", chunk_text("5ce7fa290e88ef7c") or extract_clause(r94, "5.2.1.1"))
    add("fac_038", chunk_text("5ce7fa290e88ef7c") or extract_clause(r94, "5.2.1.1"))
    add("fac_004", chunk_text("df67ba8981fd84e2") or extract_clause(r94, "1 Scope") or extract_around(r94, "This Regulation applies to vehicles of category M1"))
    add("fac_015", chunk_text("df67ba8981fd84e2"))
    add("fac_005", chunk_text("41a0a38e225fb10d") or extract_around(r94, '"Protective system" means'))
    add("fac_006", chunk_text("8af2062c6e804768") or extract_clause(r94, "5.2.1.6"))
    add("fac_007", chunk_text("6253a25685b79356") or extract_around(r94, "The application for approval of a vehicle type"))
    add(
        "fac_008",
        extract_around(r94, "impact speed")
        or extract_around(r94, "56 +0/-1 km/h")
        or extract_around(r94, "56 km/h"),
    )
    add("fac_009", chunk_text("e740f06e73e45aa7") or extract_around(r94, "Protection against electrical shock"))
    add("fac_010", chunk_text("ad557cd59ba4d53c") or extract_around(r94, "Hybrid III fiftieth percentile"))
    add("fac_011", chunk_text("6253a25685b79356"))
    add(
        "fac_012",
        chunk_text("544b51313b981e4f")
        or extract_around(r94, "The performance criteria recorded"),
    )
    add("fac_013", chunk_text("efaa3fe01ca6c6fd") or chunk_text("4ee703952b1f2621") or extract_around(r94, '"H point" means'))
    add("fac_014", chunk_text("a0db5ed6c55fcaa6") or chunk_text("eaf9979fb53da3ec") or extract_clause(r94, "5.2.1.5"))

    # R95
    add("fac_016", chunk_text("c28622fd56ef335a") or extract_around(r95, "This Regulation applies to vehicles of category M1"))
    add(
        "fac_017",
        extract_around(r95, "Thoracic Trauma Index")
        or extract_around(r95, "TTI"),
    )
    add(
        "fac_018",
        extract_around(r95, "Viscous Criterion")
        or chunk_text("72ba0419acedbfb1"),
    )
    add("fac_019", chunk_text("d3cbce2a5b63f997") or extract_around(r95, "mobile deformable barrier"))
    add("fac_020", chunk_text("751d4dc6271ede2f") or extract_around(r95, "ES-2"))
    add(
        "fac_021",
        chunk_text("f4047241b49629e7")
        or extract_around(r95, "The vehicle shall undergo a test in accordance with Annex 4"),
    )
    add(
        "fac_022",
        chunk_text("a52591a71dd986d4")
        or extract_around(r95, '"Vehicle type" means'),
    )
    add(
        "fac_039",
        extract_around(r95, "Thoracic Trauma Index")
        or extract_around(r95, "TTI shall not exceed")
        or chunk_text("f4047241b49629e7"),
    )

    # R16
    add("fac_023", chunk_text("36377415e33df408") or chunk_text("59b670e941ad2a99"))
    add(
        "fac_024",
        extract_clause(r16, "6.2.5.3.1")
        or extract_around(r16, "6.2.5.3.1")
        or chunk_text("9faf1ccbc612c5b3"),
    )
    add("fac_042", extract_clause(r16, "6.2.5.3.1") or extract_around(r16, "6.2.5.3.1") or chunk_text("9faf1ccbc612c5b3"))
    add(
        "fac_025",
        extract_around(r16, "2.40")
        or extract_around(r16, "Safety-belt reminder")
        or extract_around(r16, "safety-belt reminder"),
    )
    add("fac_043", extract_around(r16, "2.40") or extract_around(r16, "Safety-belt reminder"))
    add("fac_026", extract_around(r16, "buckle") or extract_around(r16, "Buckle"))
    add("fac_027", chunk_text("9faf1ccbc612c5b3") or extract_around(r16, "Emergency locking retractor"))
    add("fac_028", extract_around(r16, "approval mark") or extract_around(r16, "Approval mark"))
    add(
        "fac_029",
        extract_around(r16, "child restraint")
        or chunk_text("0c299f28670c4b02"),
    )

    # R129
    add(
        "fac_030",
        extract_around(r129, "Enhanced Child Restraint System")
        or chunk_text("f32e086b605fb48f"),
    )
    add(
        "fac_031",
        extract_around(r129, "i-Size")
        or chunk_text("081fc31399157079"),
    )
    add("fac_032", extract_around(r129, "stature") or extract_around(r129, "height range") or chunk_text("23510f6f0a7ba475"))
    add("fac_033", extract_around(r129, "ISOFIX") or chunk_text("081fc31399157079"))
    add("fac_034", extract_around(r129, "side impact") or extract_around(r129, "lateral"))
    add("fac_035", extract_around(r129, "Regulation No. 44") or extract_around(r129, "R.44") or extract_around(r129, "No. 44"))
    add("fac_036", extract_around(r129, "dynamic test") or chunk_text("f32e086b605fb48f"))
    add("fac_037", extract_around(r129, "dummy") or extract_around(r129, "dummies"))

    # --- cross_regulation ---
    add("xrg_001", chunk_text("df67ba8981fd84e2"))
    add(
        "xrg_002",
        chunk_text("df67ba8981fd84e2"),
        chunk_text("c28622fd56ef335a")
        or extract_around(r95, "This Regulation applies to vehicles of category M1"),
        extract_around(r94, "frontal collision") or extract_around(r94, "frontal impact"),
        extract_around(r95, "lateral collision") or extract_around(r95, "side impact"),
    )
    add(
        "xrg_003",
        extract_around(r94, "frontal") or chunk_text("df67ba8981fd84e2"),
        extract_around(r95, "lateral") or chunk_text("c28622fd56ef335a"),
    )
    add("xrg_015", *refs.get("xrg_002", []))
    add(
        "xrg_004",
        extract_around(r16, "safety-belt") or chunk_text("59b670e941ad2a99"),
        chunk_text("df67ba8981fd84e2"),
    )
    add(
        "xrg_005",
        extract_around(r16, "safety-belt") or chunk_text("59b670e941ad2a99"),
        extract_around(r94, "protective system") or chunk_text("41a0a38e225fb10d"),
    )
    add(
        "xrg_006",
        chunk_text("1faa4e17b012c1ab"),
        chunk_text("9a19875b521d408a"),
    )
    add(
        "xrg_007",
        chunk_text("e740f06e73e45aa7") or extract_around(r94, "Protection against electrical shock"),
        extract_around(r95, "electrical") or chunk_text("f4047241b49629e7"),
    )
    add("xrg_008", *refs.get("xrg_007", []))
    add("xrg_009", chunk_text("c28622fd56ef335a") or extract_around(r95, "This Regulation applies to vehicles of category M1"))
    add("xrg_010", chunk_text("df67ba8981fd84e2"))
    add(
        "xrg_011",
        extract_around(r94, "frontal") or chunk_text("df67ba8981fd84e2"),
        extract_around(r95, "lateral") or chunk_text("c28622fd56ef335a"),
        chunk_text("5ce7fa290e88ef7c"),
        chunk_text("f4047241b49629e7"),
    )
    add(
        "xrg_012",
        extract_around(r16, "retractor") or chunk_text("9faf1ccbc612c5b3"),
        chunk_text("ad557cd59ba4d53c") or extract_around(r94, "Hybrid III"),
    )
    add(
        "xrg_013",
        chunk_text("930c351518a8769b"),
        extract_around(r95, "Rib Deflection Criterion")
        or extract_around(r95, "RDC")
        or chunk_text("f4047241b49629e7"),
    )
    add(
        "xrg_016",
        chunk_text("5ce7fa290e88ef7c"),
        chunk_text("930c351518a8769b"),
        chunk_text("f4047241b49629e7"),
    )
    # xrg_014 compares to FMVSS — reference is R94 side only; FMVSS not in corpus
    add(
        "xrg_014",
        chunk_text("5ce7fa290e88ef7c")
        or extract_clause(r94, "5.2.1.1"),
    )

    return refs


def main() -> None:
    goldens = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {g["id"]: g for g in goldens}
    refs = build_refs()

    accepted = 0
    rejected = 0
    skipped_cats = 0
    report: list[str] = []

    for case_id, texts in sorted(refs.items()):
        case = by_id.get(case_id)
        if not case:
            rejected += 1
            report.append(f"MISS id {case_id}")
            continue
        if case.get("category") not in {"factual_lookup", "cross_regulation"}:
            skipped_cats += 1
            continue
        contains = case.get("expected_answer_contains") or []
        joined = "\n\n".join(texts)
        if not spot_check(joined, contains):
            # keep only if no contains tokens, or soft-fail with note
            missing = [
                t
                for t in contains
                if str(t).strip()
                and str(t).strip().lower() not in joined.lower()
                and not (
                    re.fullmatch(r"[\d.]+", str(t).strip())
                    and str(t).strip().replace(",", "")
                    in re.sub(r"[,\s]", "", joined.lower())
                )
            ]
            if missing:
                rejected += 1
                report.append(
                    f"REJECT {case_id}: missing {missing} in refs "
                    f"({joined[:120]!r}...)"
                )
                continue
        case["ground_truth_reference_chunks"] = texts
        accepted += 1
        report.append(f"OK {case_id}: {len(texts)} chunk(s), {len(joined)} chars")

    # Rewrite golden set preserving order
    GOLDEN.write_text(
        "\n".join(json.dumps(g, ensure_ascii=False) for g in goldens) + "\n",
        encoding="utf-8",
    )

    empty = sum(
        1
        for g in goldens
        if not g.get("ground_truth_reference_chunks")
    )
    fac_filled = sum(
        1
        for g in goldens
        if g.get("category") == "factual_lookup" and g.get("ground_truth_reference_chunks")
    )
    xrg_filled = sum(
        1
        for g in goldens
        if g.get("category") == "cross_regulation" and g.get("ground_truth_reference_chunks")
    )
    print("\n".join(report))
    print("---")
    print(f"accepted={accepted} rejected={rejected} skipped_other_cat={skipped_cats}")
    print(f"golden empty refs remaining: {empty}/{len(goldens)}")
    print(f"factual_lookup filled: {fac_filled}/43")
    print(f"cross_regulation filled: {xrg_filled}/16")


if __name__ == "__main__":
    main()
