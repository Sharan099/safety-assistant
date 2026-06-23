"""Targeted enrichment for sparse UN R14 annex table sections."""

from __future__ import annotations

import re
from pathlib import Path

_ANNEX6_START = re.compile(r"^Annex\s+6\s*$", re.I | re.M)
_ANNEX6_APP1 = re.compile(r"^Annex\s+6\s*-\s*Appendix\s+1", re.I | re.M)


def extract_annex6_body(markdown_text: str) -> str | None:
    """Pull Annex 6 seating-position / anchorage-count table from markdown."""
    m = _ANNEX6_START.search(markdown_text)
    if not m:
        return None
    start = m.start()
    m_end = _ANNEX6_APP1.search(markdown_text, start + 1)
    end = m_end.start() if m_end else start + 4000
    block = markdown_text[start:end].strip()
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    # Drop page headers / lone page numbers
    cleaned = [
        ln for ln in lines
        if not re.match(r"^E/ECE/", ln)
        and not re.match(r"^\d{1,3}$", ln)
        and ln.lower() != "annex 6"
    ]
    body = "\n".join(cleaned)
    if len(body.split()) < 40:
        return None
    return (
        "UN R14 Annex 6 — Minimum number of anchorage points and location of lower "
        f"anchorages (vehicle category seating table):\n{body}"
    )


def enrich_annex_chunks(chunks: list[dict], md_path: Path) -> list[dict]:
    """Patch near-empty Annex 6 chunks with full table text from markdown source."""
    if md_path.stem.upper() not in ("UN_R14",):
        return chunks
    try:
        md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return chunks
    annex_body = extract_annex6_body(md_text)
    if not annex_body:
        return chunks

    out: list[dict] = []
    for c in chunks:
        if c.get("clause") == "Annex 6" or (c.get("section_title") or "").strip() == "Annex 6":
            if len((c.get("text") or "").split()) < 40:
                patched = dict(c)
                header = patched.get("text", "").split("\n\n")[0]
                patched["text"] = f"{header}\n\n{annex_body}"
                patched["word_count"] = len(patched["text"].split())
                patched["has_vehicle_classes"] = True
                patched["has_belt_system"] = True
                patched["has_requirements"] = True
                patched["annex_table_enriched"] = True
                out.append(patched)
                continue
        out.append(c)
    return out
