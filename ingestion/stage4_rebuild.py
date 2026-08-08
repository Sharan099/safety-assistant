"""Stage 4 — full corpus rebuild into Qdrant from Docling exports.

Runs: remediate → chunk → Stage-2 figure describe → enrich → embed → upsert
for R94 / R95 / R16 / R129. Skips re-Docling when ``*.docling.json`` exists.
LightOnOCR is off by default here (caption remediation already applied); set
``STAGE4_AUTO_LIGHTON=1`` to re-enable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DOCLING_DIR = ROOT / "data" / "docling"
PDF_DIR = ROOT / "data" / "pdfs"

REGULATIONS: tuple[tuple[str, str, str], ...] = (
    ("UN_R94", "UN-ECE-R94", "Rev.3"),
    ("UN_R95", "UN-ECE-R95", "Rev.3"),
    ("UN_R16", "UN-ECE-R16", "Rev.10"),
    ("UN_R129", "UN-ECE-R129", "Rev.4"),
)


def rebuild_one(
    stem: str,
    regulation_id: str,
    revision: str,
    *,
    skip_upsert: bool = False,
) -> dict[str, Any]:
    load_dotenv()
    from ingestion.chunk import chunk_document
    from ingestion.describe_figures import describe_and_chunk_figures
    from ingestion.embed_upsert import upsert_chunks
    from ingestion.enrich import enrich_chunks
    from ingestion.extract import extract_from_docling, load_docling_json, remediate_clause_as_caption
    from ingestion.parse import parse_pdf

    t0 = time.perf_counter()
    json_path = DOCLING_DIR / f"{stem}.docling.json"
    pdf_path = PDF_DIR / f"{stem}.pdf"
    DOCLING_DIR.mkdir(parents=True, exist_ok=True)

    if json_path.is_file():
        doc = load_docling_json(json_path)
        source = "docling_json"
    else:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Need {json_path} or {pdf_path}")
        doc = parse_pdf(pdf_path, export_dir=DOCLING_DIR)
        source = "pdf_parse"

    rem = remediate_clause_as_caption(doc)
    extract = extract_from_docling(
        doc,
        document_id=regulation_id,
        regulation_id=regulation_id,
        revision=revision,
        source_path=str(pdf_path),
        pdf_path=pdf_path if pdf_path.is_file() else None,
    )
    extract_path = DOCLING_DIR / f"{stem}.extract.json"
    extract_path.write_text(extract.model_dump_json(indent=2), encoding="utf-8")

    chunks = chunk_document(doc, regulation_id=regulation_id, revision=revision)
    fig_chunks = describe_and_chunk_figures(
        regulation_id=regulation_id,
        revision=revision,
        extract=extract,
    )
    chunks = list(chunks) + list(fig_chunks)
    enriched = enrich_chunks(chunks)

    upserted = 0
    if not skip_upsert:
        upserted = upsert_chunks(enriched, delete_existing=True)

    elapsed = time.perf_counter() - t0
    report = {
        "stem": stem,
        "regulation_id": regulation_id,
        "revision": revision,
        "source": source,
        "remediation_actions": len(rem),
        "extract_elements": len(extract.elements),
        "chunk_count": len(enriched),
        "clause_chunks": sum(1 for c in enriched if c.content_type == "clause"),
        "table_chunks": sum(1 for c in enriched if c.content_type == "table"),
        "figure_chunks": sum(1 for c in enriched if c.content_type == "figure"),
        "upserted": upserted,
        "elapsed_s": round(elapsed, 2),
    }
    logger.info("Stage4 %s → %s", regulation_id, report)
    return report


def rebuild_all(*, skip_upsert: bool = False) -> dict[str, Any]:
    """Wipe is assumed already done (Stage 0). Rebuild all four regulations."""
    load_dotenv()
    from api.cache_version import bump_cache_version
    from ingestion.wipe import wipe_qdrant_collection

    # Ensure empty hybrid collection at current embedding dim.
    wipe_qdrant_collection()

    t0 = time.perf_counter()
    per_reg: list[dict[str, Any]] = []
    for stem, rid, rev in REGULATIONS:
        per_reg.append(rebuild_one(stem, rid, rev, skip_upsert=skip_upsert))

    if not skip_upsert:
        bump_cache_version()

    expected = {r["regulation_id"]: r["chunk_count"] for r in per_reg}
    report = {
        "ok": True,
        "regulations": per_reg,
        "expected_chunk_counts": expected,
        "total_chunks": sum(r["chunk_count"] for r in per_reg),
        "total_elapsed_s": round(time.perf_counter() - t0, 2),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    }
    out = ROOT / "data" / "stage4_rebuild_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    skip = os.getenv("STAGE4_SKIP_UPSERT", "").strip().lower() in {"1", "true", "yes"}
    report = rebuild_all(skip_upsert=skip)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
