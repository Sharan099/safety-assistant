"""PDF path registry + Qdrant citation lookup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from qdrant_client.http import models as qm

from ingestion.embed_upsert import DEFAULT_COLLECTION, get_qdrant_client

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = Path(os.getenv("PDF_DIR") or ROOT / "data" / "pdfs")

# regulation_id → filename under data/pdfs/
REGULATION_PDFS: dict[str, str] = {
    "UN-ECE-R94": "UN_R94.pdf",
    "UN-ECE-R95": "UN_R95.pdf",
    "UN-ECE-R16": "UN_R16.pdf",
    "UN-ECE-R129": "UN_R129.pdf",
    "R94": "UN_R94.pdf",
    "R95": "UN_R95.pdf",
    "R16": "UN_R16.pdf",
    "R129": "UN_R129.pdf",
}


def resolve_pdf(regulation_id: str) -> Path:
    key = (regulation_id or "").strip()
    filename = REGULATION_PDFS.get(key) or REGULATION_PDFS.get(key.upper())
    if not filename:
        # Allow UN-ECE-R94 style fuzzy
        for rid, name in REGULATION_PDFS.items():
            if rid.lower() in key.lower() or key.lower() in rid.lower():
                filename = name
                break

    candidates: list[Path] = []
    if filename:
        candidates.append(PDF_DIR / filename)

    # Uploaded / auto-named copies (data/pdfs + data/raw).
    short = key.replace("UN-ECE-", "").replace(" ", "")
    for folder in (PDF_DIR, ROOT / "data" / "raw"):
        candidates.extend(
            [
                folder / f"UN_{short}.pdf",
                folder / f"{key}.pdf",
                folder / f"{short}.pdf",
            ]
        )
        if folder.is_dir():
            for p in folder.glob("*.pdf"):
                stem = p.stem.upper().replace("-", "").replace("_", "")
                needle = key.upper().replace("-", "").replace("_", "")
                if needle and needle in stem:
                    candidates.append(p)

    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve() if path.exists() else path
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path

    raise FileNotFoundError(f"No PDF mapped for regulation_id={regulation_id!r}")


def lookup_citation(chunk_id: str) -> dict[str, Any] | None:
    """Return page_number + bounding_box (+ metadata) for a chunk_id."""
    chunk_id = (chunk_id or "").strip()
    if not chunk_id:
        return None
    client = get_qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        return None

    # Exact chunk_id match
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qm.Filter(
            must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=chunk_id))]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points and chunk_id.startswith("expanded::"):
        # Expanded parent ids: match section_id
        section_id = chunk_id[len("expanded::") :]
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="section_id", match=qm.MatchValue(value=section_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

    if not points:
        return None
    payload = points[0].payload or {}
    bbox = list(payload.get("bounding_box") or [])
    return {
        "chunk_id": payload.get("chunk_id") or chunk_id,
        "regulation_id": payload.get("regulation_id") or "",
        "section_number": payload.get("section_number") or "",
        "section_title": payload.get("section_title") or "",
        "section_id": payload.get("section_id") or "",
        "page_number": payload.get("page_number"),
        "bounding_box": bbox,
        "coord_origin": "BOTTOMLEFT",  # Docling default for PDF provenance
        "text": payload.get("text") or "",
        "citation": _citation_from_payload(payload),
    }


def _citation_from_payload(payload: dict[str, Any]) -> str:
    reg = payload.get("regulation_id") or "?"
    sec = payload.get("section_number") or "?"
    page = payload.get("page_number")
    page_s = page if page is not None else "?"
    return f"[{reg} §{sec}, p.{page_s}]"
