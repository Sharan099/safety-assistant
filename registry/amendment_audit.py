"""Front-matter vs filename amendment audit (FR-16)."""

from __future__ import annotations

import re
from typing import Any

from regulation_discovery.registry.version_parser import parse_document_metadata_from_filename


def _normalize_series(value: str | None) -> str | None:
    if not value:
        return None
    m = re.search(r"(\d+)", str(value))
    if not m:
        return str(value).strip()
    return f"{int(m.group(1)):02d}"


def audit_amendment_metadata(
    extracted_metadata: dict[str, Any],
    filename: str,
) -> dict[str, Any]:
    """
    Compare amendment/series from document text vs filename.
    Returns flags merged into metadata for registry persistence.
    """
    from_text = extracted_metadata.get("amendment")
    file_meta = parse_document_metadata_from_filename(filename)
    series_num = file_meta.get("series_number")
    from_filename = f"{series_num:02d} Series" if series_num is not None else None

    norm_text = _normalize_series(from_text)
    norm_file = _normalize_series(from_filename)
    mismatch = bool(norm_text and norm_file and norm_text != norm_file)

    return {
        "amendment_from_text": from_text,
        "amendment_from_filename": from_filename,
        "amendment_mismatch": mismatch,
        "amendment_audit": {
            "from_text": from_text,
            "from_filename": from_filename,
            "mismatch": mismatch,
        },
    }
