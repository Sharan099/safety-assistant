"""Staging manifest for human-dropped PDFs with non-standard filenames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MANIFEST_NAMES = ("manifest.yaml", "manifest.yml", "manifest.json")


def _normalize_reg_code(code: str) -> str:
    rid = str(code).strip().upper().replace(" ", "_")
    if rid.startswith("UN_R"):
        return rid
    if rid.startswith("R") and rid[1:].isdigit():
        return f"UN_{rid}"
    if rid.isdigit():
        return f"UN_R{rid}"
    return rid


def load_staging_manifest(manifest_path: Path | None, staging_dir: Path) -> dict[str, dict[str, str]]:
    """
    Load per-file overrides keyed by staging filename (case-insensitive).

    Returns mapping: lower(filename) -> {"regulation_code": "UN_R16", "amendment": "Base", ...}
    """
    path = manifest_path
    if path is None:
        for name in DEFAULT_MANIFEST_NAMES:
            candidate = staging_dir / name
            if candidate.exists():
                path = candidate
                break
    if path is None or not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        data = yaml.safe_load(raw) or {}

    entries: list[dict[str, Any]] = []
    if isinstance(data, dict):
        entries = data.get("files") or data.get("mappings") or []
    elif isinstance(data, list):
        entries = data

    mapping: dict[str, dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename") or entry.get("file")
        reg_code = entry.get("regulation_code") or entry.get("regulation") or entry.get("reg_code")
        if not filename or not reg_code:
            continue
        key = Path(str(filename)).name.lower()
        mapping[key] = {
            "regulation_code": _normalize_reg_code(str(reg_code)),
            "amendment": str(entry.get("amendment") or entry.get("series") or "").strip(),
            "source_url": str(entry.get("source_url") or "").strip(),
        }
    return mapping


def gather_staging_pdfs(staging_dir: Path) -> list[Path]:
    """Collect every PDF directly under staging (non-recursive; skip manifest files)."""
    if not staging_dir.is_dir():
        return []
    manifest_names = {n.lower() for n in DEFAULT_MANIFEST_NAMES}
    pdfs = []
    for path in sorted(staging_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".pdf":
            continue
        if path.name.lower() in manifest_names:
            continue
        pdfs.append(path)
    return pdfs
