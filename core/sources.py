"""Pinned UNECE knowledge sources for passive safety RAG."""

from __future__ import annotations

from pathlib import Path

from app.config import settings

SOURCES = [
    {
        "regulation_code": "UN_R94",
        "title": "UN Regulation No. 94 — Protection of occupants in frontal collision",
        "filename": "UN_R94.pdf",
        "topic": "Frontal impact, injury criteria (HIC, ThCC, femur)",
    },
    {
        "regulation_code": "UN_R95",
        "title": "UN Regulation No. 95 — Protection of occupants in side impact",
        "filename": "UN_R95.pdf",
        "topic": "Side barrier intrusion, chest/abdomen limits",
    },
    {
        "regulation_code": "UN_R16",
        "title": "UN Regulation No. 16 — Safety belts and restraint systems",
        "filename": "UN_R16.pdf",
        "topic": "Seat belts, anchorage, retractor locking",
    },
    {
        "regulation_code": "UN_R129",
        "title": "UN Regulation No. 129 — Enhanced child restraint systems (i-Size)",
        "filename": "UN_R129.pdf",
        "topic": "Child restraints, height classes, ISOFIX",
    },
]


def pdf_dir() -> Path:
    return settings.PDF_DIR


def source_paths() -> list[Path]:
    return [pdf_dir() / s["filename"] for s in SOURCES]


def sources_payload() -> list[dict]:
    out = []
    for s in SOURCES:
        path = pdf_dir() / s["filename"]
        out.append({**s, "available": path.is_file(), "path": str(path)})
    return out
