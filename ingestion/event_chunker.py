"""Event-based chunking for engineering / historical / rating documents (Part C)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from ingestion.synthetic_chunker import (
    _EVENT_SECTIONS,
    _TABLE_RE,
    chunk_synthetic_events,
    is_synthetic_markdown,
)

_REFUSAL_LICENSE = "review_needed"

_ENGINEERING_SECTIONS = _EVENT_SECTIONS + (
    "Design rationale",
    "Risk assessment",
    "Kinematics assessment",
    "Revalidation scope",
    "Checklist",
)


def license_allows_ingest(license_status: str | None) -> bool:
    return (license_status or "public_domain") != _REFUSAL_LICENSE


def is_event_document(md_path: Path, meta: dict, regulation: str) -> bool:
    if is_synthetic_markdown(md_path, meta):
        return True
    kind = (meta.get("document_kind") or meta.get("source_type") or "").lower()
    if kind in ("test_report", "rca", "crash_investigation", "oem_standard", "rating_protocol"):
        return True
    if regulation.startswith("PROG_X") or regulation in ("EURO_NCAP", "CAE_REFERENCE", "SAFETY_REFERENCE"):
        return True
    return False


def chunk_event_document(
    md_path: Path,
    text: str,
    meta: dict,
    *,
    make_chunk: Callable[..., dict],
    regulation: str,
    file_slug: str,
    pdf_name: str,
) -> list[dict]:
    """Bundle observation + measurement + judgement into atomic event chunks."""
    license_status = meta.get("license_status", "public_domain")
    if not license_allows_ingest(license_status):
        raise ValueError(
            f"Refusing ingest for {md_path.name}: license_status=review_needed "
            "(requires human sign-off)"
        )
    return chunk_synthetic_events(
        md_path, text, meta,
        make_chunk=make_chunk,
        regulation=regulation,
        file_slug=file_slug,
        pdf_name=pdf_name,
    )
