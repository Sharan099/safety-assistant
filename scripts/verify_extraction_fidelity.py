#!/usr/bin/env python3
"""
Independent extraction-fidelity audit (Part A).

Re-OCRs pilot PDFs with a pipeline DISTINCT from the production PaddleOCR ingest
(default: Docling + table structure; optional: Tesseract on rendered page images)
and diffs numeric values, table structure, and cross-references against the
ingested markdown under output/markdown/.

Outputs: output/extraction_fidelity_report.json

Gate: zero numeric-value discrepancies with severity=blocking before Part B work.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from config import CORPUS_DIR, MARKDOWN_DIR, OUTPUT_DIR  # noqa: E402

REPORT_PATH = OUTPUT_DIR / "extraction_fidelity_report.json"

# ── numeric extraction ──────────────────────────────────────────────

_NUM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("dan", re.compile(r"(\d{1,4}(?:[.,]\d+)?)\s*(?:±|\+/-)?\s*(\d{1,3})?\s*daN", re.I)),
    ("kn", re.compile(r"(\d{1,4}(?:[.,]\d+)?)\s*(?:±|\+/-)?\s*(\d{1,3})?\s*kN", re.I)),
    ("mm", re.compile(r"(\d{1,4}(?:[.,]\d+)?)\s*(?:±|\+/-)?\s*(\d{1,3})?\s*mm\b", re.I)),
    ("kg", re.compile(r"(\d{1,4}(?:[.,]\d+)?)\s*(?:±|\+/-)?\s*(\d{1,3})?\s*kg\b", re.I)),
    ("degree", re.compile(r"(\d{1,3})\s*°", re.I)),
    ("degree_range", re.compile(r"(\d{1,3})\s*°\s*[-–]\s*(\d{1,3})\s*°", re.I)),
    ("percent", re.compile(r"(\d{1,3}(?:[.,]\d+)?)\s*%", re.I)),
    ("time_s", re.compile(r"(\d+(?:[.,]\d+)?)\s*second", re.I)),
    ("time_s_short", re.compile(r"not less than\s+(\d+(?:[.,]\d+)?)\s*s\b", re.I)),
    ("hz", re.compile(r"(\d+(?:[.,]\d+)?)\s*Hz", re.I)),
    ("cycles", re.compile(r"(\d{1,3}(?:,\d{3})*)\s*(?:cycles)?", re.I)),
]


def _to_float(num: str) -> float:
    return float(num.replace(",", "").replace(" ", ""))


def extract_numeric_tokens(text: str) -> list[dict[str, Any]]:
    """Pull normalized numeric tokens from text for set comparison."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, pat in _NUM_PATTERNS:
        for m in pat.finditer(text):
            if kind == "degree_range":
                key = f"{kind}:{m.group(1)}-{m.group(2)}"
                val = (float(m.group(1)), float(m.group(2)))
            elif kind == "dan":
                main = _to_float(m.group(1))
                tol = _to_float(m.group(2)) if m.group(2) else None
                key = f"dan:{main}:{tol}"
                val = main
            else:
                g = m.group(1)
                if not g or not re.search(r"\d", g):
                    continue
                val = _to_float(g)
                key = f"{kind}:{val}"
            if key in seen:
                continue
            seen.add(key)
            found.append({
                "kind": kind,
                "value": val,
                "raw": m.group(0).strip(),
                "key": key,
            })
    return found


def numeric_set(tokens: list[dict[str, Any]]) -> set[str]:
    return {t["key"] for t in tokens}


def diff_numeric_sets(
    reference: set[str],
    candidate: set[str],
    *,
    page: int | None,
    regulation: str,
) -> list[dict[str, Any]]:
    """Values in markdown missing from independent OCR = potential extraction loss."""
    missing = reference - candidate
    extra = candidate - reference
    out: list[dict[str, Any]] = []
    for key in sorted(missing):
        out.append({
            "severity": "blocking",
            "type": "numeric_missing_in_independent_ocr",
            "regulation": regulation,
            "page": page,
            "key": key,
            "detail": f"Markdown numeric {key!r} not found in independent OCR on same page",
        })
    for key in sorted(extra):
        out.append({
            "severity": "warning",
            "type": "numeric_extra_in_independent_ocr",
            "regulation": regulation,
            "page": page,
            "key": key,
            "detail": f"Independent OCR has {key!r} not present in ingested markdown page",
        })
    return out


# ── page splitting ────────────────────────────────────────────────

_PAGE_HDR = re.compile(r"^## Page (\d+)\s*$", re.M)


def split_by_page(text: str) -> dict[int, str]:
    pages: dict[int, str] = {}
    matches = list(_PAGE_HDR.finditer(text))
    if not matches:
        pages[1] = text
        return pages
    for i, m in enumerate(matches):
        pnum = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pages[pnum] = text[start:end]
    return pages


# ── table / cross-ref checks ──────────────────────────────────────

@dataclass
class TableSpec:
    regulation: str
    name: str
    start_pattern: str
    end_pattern: str | None = None
    must_contain: tuple[str, ...] = ()
    symbol_markers: tuple[str, ...] = ()
    min_pipe_rows: int = 3


@dataclass
class CrossRefSpec:
    regulation: str
    clause: str
    pattern: str
    description: str


REQUIRED_TABLES: list[TableSpec] = [
    TableSpec(
        regulation="UN_R14",
        name="Annex 6 anchorage-count table",
        start_pattern=r"^Annex\s+6\s*$",
        end_pattern=r"^Annex\s+6\s*-\s*Appendix\s+1",
        must_contain=("M1", "M3", "N1", "Vehicle category"),
        symbol_markers=("Ø", "*", "╬", ""),
        min_pipe_rows=4,
    ),
    TableSpec(
        regulation="UN_R14",
        name="Annex 6 Appendix 1 angle table",
        start_pattern=r"^Annex\s+6\s*-\s*Appendix\s+1",
        end_pattern=r"^Annex\s+7",
        must_contain=("45°", "80°", "M1", "angle"),
        symbol_markers=(),
        min_pipe_rows=4,
    ),
    TableSpec(
        regulation="UN_R16",
        name="§7.4.1.6.3 abrasion procedure table",
        start_pattern=r"7\.4\.1\.6\.3\.",
        end_pattern=r"7\.4\.1\.6\.4\.",
        must_contain=("Procedure 1", "2.5", "5,000", "Hz"),
        symbol_markers=(),
        min_pipe_rows=3,
    ),
]

CROSS_REFS: list[CrossRefSpec] = [
    CrossRefSpec(
        regulation="UN_R14",
        clause="5.3.1.1",
        pattern=r"UN Regulation No\.\s*16",
        description="R14 harness belt anchorages reference R16 approval",
    ),
    CrossRefSpec(
        regulation="UN_R16",
        clause="8.2.1",
        pattern=r"UN Regulation No\.\s*14",
        description="R16 general requirements reference R14 anchorages",
    ),
]


def _extract_block(md: str, spec: TableSpec) -> str | None:
    m = re.search(spec.start_pattern, md, re.I | re.M)
    if not m:
        return None
    start = m.start()
    if spec.end_pattern:
        m_end = re.search(spec.end_pattern, md[start + 1 :], re.I | re.M)
        end = start + 1 + m_end.start() if m_end else start + 6000
    else:
        end = start + 6000
    return md[start:end]


def _count_pipe_table_rows(block: str) -> int:
    return sum(1 for ln in block.splitlines() if ln.strip().startswith("|") and "|" in ln[1:])


def check_table_structure(md: str, spec: TableSpec) -> list[dict[str, Any]]:
    block = _extract_block(md, spec)
    issues: list[dict[str, Any]] = []
    if block is None:
        return [{
            "severity": "blocking",
            "type": "table_section_missing",
            "regulation": spec.regulation,
            "table": spec.name,
            "detail": f"Could not locate section for {spec.name}",
        }]
    missing = [t for t in spec.must_contain if t not in block]
    if missing:
        issues.append({
            "severity": "blocking",
            "type": "table_content_missing",
            "regulation": spec.regulation,
            "table": spec.name,
            "detail": f"Missing expected tokens: {missing}",
        })
    for sym in spec.symbol_markers:
        if sym not in block:
            issues.append({
                "severity": "warning",
                "type": "table_symbol_missing",
                "regulation": spec.regulation,
                "table": spec.name,
                "detail": f"Symbol {sym!r} not found in table block",
            })
    pipe_rows = _count_pipe_table_rows(block)
    if pipe_rows < spec.min_pipe_rows:
        issues.append({
            "severity": "blocking",
            "type": "table_not_structured",
            "regulation": spec.regulation,
            "table": spec.name,
            "detail": (
                f"Table degraded to linearized text (pipe rows={pipe_rows}, "
                f"need >={spec.min_pipe_rows}). Re-extract with layout/table model."
            ),
        })
    return issues


def check_cross_references(md: str, spec: CrossRefSpec) -> list[dict[str, Any]]:
    # Locate clause neighborhood
    clause_re = re.compile(rf"{re.escape(spec.clause)}\s*\.?", re.M)
    m = clause_re.search(md)
    if not m:
        return [{
            "severity": "warning",
            "type": "clause_anchor_missing",
            "regulation": spec.regulation,
            "clause": spec.clause,
            "detail": f"Clause {spec.clause} anchor not found for cross-ref check",
        }]
    window = md[m.start() : m.start() + 1200]
    if not re.search(spec.pattern, window, re.I):
        return [{
            "severity": "blocking",
            "type": "cross_reference_missing",
            "regulation": spec.regulation,
            "clause": spec.clause,
            "detail": spec.description,
        }]
    return []


# ── independent OCR backends ───────────────────────────────────────

def ocr_pdf_docling(pdf_path: Path) -> str:
    from ingestion.docling_converter import convert_pdf_docling

    out = MARKDOWN_DIR / f"_fidelity_audit_{pdf_path.stem}.md"
    convert_pdf_docling(pdf_path, out)
    return out.read_text(encoding="utf-8", errors="ignore")


def ocr_pdf_tesseract_pages(pdf_path: Path, *, max_pages: int | None = None) -> dict[int, str]:
    """Render pages with PyMuPDF; OCR with Tesseract CLI (independent of Paddle)."""
    import fitz

    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract CLI not found on PATH")

    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pytesseract and Pillow required for tesseract mode") from exc

    doc = fitz.open(pdf_path)
    pages: dict[int, str] = {}
    limit = len(doc) if max_pages is None else min(len(doc), max_pages)
    for i in range(limit):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="eng")
        pages[i + 1] = text
    doc.close()
    return pages


def compare_page_numerics(
    md_pages: dict[int, str],
    ocr_pages: dict[int, str],
    regulation: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    all_pages = sorted(set(md_pages) | set(ocr_pages))
    for p in all_pages:
        md_toks = extract_numeric_tokens(md_pages.get(p, ""))
        ocr_toks = extract_numeric_tokens(ocr_pages.get(p, ""))
        md_set = numeric_set(md_toks)
        ocr_set = numeric_set(ocr_toks)
        if not md_set and not ocr_set:
            continue
        issues.extend(
            diff_numeric_sets(md_set, ocr_set, page=p, regulation=regulation)
        )
    return issues


# ── report assembly ───────────────────────────────────────────────

@dataclass
class FidelityReport:
    generated_at: str
    mode: str
    independent_engine: str
    regulations: list[str] = field(default_factory=list)
    discrepancies: list[dict[str, Any]] = field(default_factory=list)
    blocking_numeric_count: int = 0
    blocking_table_count: int = 0
    numeric_gate_pass: bool = False
    gate_pass: bool = False
    summary: str = ""

    def finalize(self) -> None:
        blocking = [d for d in self.discrepancies if d.get("severity") == "blocking"]
        self.blocking_numeric_count = sum(
            1 for d in blocking if "numeric" in d.get("type", "")
        )
        self.blocking_table_count = sum(
            1 for d in blocking if "table" in d.get("type", "")
        )
        # Part B blocker: zero numeric-value discrepancies only.
        self.numeric_gate_pass = self.blocking_numeric_count == 0
        self.gate_pass = self.numeric_gate_pass
        self.summary = (
            f"{len(self.discrepancies)} discrepancies "
            f"({len(blocking)} blocking, "
            f"{self.blocking_numeric_count} numeric, "
            f"{self.blocking_table_count} table-structure); "
            f"numeric_gate={'PASS' if self.numeric_gate_pass else 'FAIL'}"
        )


def audit_regulation(
    reg_code: str,
    *,
    mode: str,
    engine: str,
    max_pages: int | None,
) -> list[dict[str, Any]]:
    pdf_path = CORPUS_DIR / "legal" / f"{reg_code}.pdf"
    md_path = MARKDOWN_DIR / f"{reg_code}.md"
    if not pdf_path.is_file() or not md_path.is_file():
        return [{
            "severity": "blocking",
            "type": "corpus_missing",
            "regulation": reg_code,
            "detail": f"Missing {pdf_path} or {md_path}",
        }]

    md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    issues: list[dict[str, Any]] = []

    # Structural checks (always)
    for spec in REQUIRED_TABLES:
        if spec.regulation != reg_code:
            continue
        issues.extend(check_table_structure(md_text, spec))
    for spec in CROSS_REFS:
        if spec.regulation != reg_code:
            continue
        issues.extend(check_cross_references(md_text, spec))

    if mode == "quick":
        # Spot-check critical numerics present in markdown (no second OCR)
        anchors = {
            "UN_R14": ["1,350", "0.2 second", "450", "675", "10°"],
            "UN_R16": ["2.5", "5,000", "0.5", "Hz"],
        }
        for token in anchors.get(reg_code, []):
            if token not in md_text:
                issues.append({
                    "severity": "blocking",
                    "type": "numeric_anchor_missing_in_markdown",
                    "regulation": reg_code,
                    "detail": f"Expected anchor token {token!r} missing from ingested markdown",
                })
        return issues

    # Full independent OCR diff
    md_pages = split_by_page(md_text)
    if engine == "tesseract":
        ocr_pages = ocr_pdf_tesseract_pages(pdf_path, max_pages=max_pages)
    else:
        ocr_text = ocr_pdf_docling(pdf_path)
        ocr_pages = split_by_page(ocr_text)

    if max_pages:
        md_pages = {k: v for k, v in md_pages.items() if k <= max_pages}
        ocr_pages = {k: v for k, v in ocr_pages.items() if k <= max_pages}

    issues.extend(compare_page_numerics(md_pages, ocr_pages, reg_code))
    return issues


def run_audit(*, mode: str = "quick", engine: str = "docling", max_pages: int | None = None) -> FidelityReport:
    from datetime import datetime, timezone

    report = FidelityReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        independent_engine=engine,
        regulations=["UN_R14", "UN_R16"],
    )
    for reg in report.regulations:
        report.discrepancies.extend(
            audit_regulation(reg, mode=mode, engine=engine, max_pages=max_pages)
        )
    report.finalize()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Extraction fidelity audit (Part A)")
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="quick=structure+cross-ref+anchors; full=independent OCR numeric diff",
    )
    parser.add_argument(
        "--engine",
        choices=("docling", "tesseract"),
        default="docling",
        help="Independent OCR engine (not PaddleOCR)",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages for full mode")
    args = parser.parse_args()

    t0 = time.perf_counter()
    report = run_audit(mode=args.mode, engine=args.engine, max_pages=args.max_pages)
    elapsed = round(time.perf_counter() - t0, 1)
    print(f"Wrote {REPORT_PATH}")
    print(f"{report.summary} ({elapsed}s)")
    for d in report.discrepancies[:20]:
        print(f"  [{d.get('severity')}] {d.get('type')}: {d.get('detail', d)}")
    if len(report.discrepancies) > 20:
        print(f"  ... and {len(report.discrepancies) - 20} more")
    return 0 if report.gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
