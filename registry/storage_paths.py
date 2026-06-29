"""Canonical storage paths and deterministic naming (FR-9, FR-10)."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

AUTHORITY_STORAGE_DIRS: dict[str, str] = {
    "UNECE": "UNECE",
    "Euro NCAP": "EuroNCAP",
    "FMVSS": "FMVSS",
    "NHTSA": "NHTSA",
    "IIHS": "IIHS",
    "EU Regulations": "EU_Regulations",
    "China C-NCAP": "China",
    "INTERNAL": "INTERNAL",
}


def project_root() -> Path:
    return Path(os.getenv("REGISTRY_ROOT", Path.cwd()))


def staging_dir(root: Path | None = None) -> Path:
    path = (root or project_root()) / "data" / "staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def quarantine_dir(root: Path | None = None) -> Path:
    path = (root or project_root()) / "data" / "quarantine"
    path.mkdir(parents=True, exist_ok=True)
    return path


def downloads_dir(root: Path | None = None) -> Path:
    """Transient legacy download area — not canonical."""
    path = (root or project_root()) / "data" / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def storage_dir(root: Path | None = None) -> Path:
    path = (root or project_root()) / "storage"
    path.mkdir(parents=True, exist_ok=True)
    return path


def authority_folder(source_type: str) -> str:
    return AUTHORITY_STORAGE_DIRS.get(source_type, re.sub(r"[^\w]+", "", source_type))


def deterministic_filename(metadata: dict) -> str:
    code = metadata.get("regulation_code", "UNKNOWN")
    amendment = metadata.get("amendment") or "Base"
    source = metadata.get("source_type", "INTERNAL")
    safe_code = re.sub(r"[^\w]+", "_", str(code)).strip("_")
    safe_amend = re.sub(r"[^\w]+", "_", str(amendment)).strip("_")
    safe_source = re.sub(r"[^\w]+", "_", str(source)).strip("_")
    return f"{safe_source}_{safe_code}_{safe_amend}.pdf"


def canonical_path(metadata: dict, root: Path | None = None) -> Path:
    folder = authority_folder(metadata.get("source_type", "INTERNAL"))
    name = deterministic_filename(metadata)
    return storage_dir(root) / folder / name


def promote_to_storage(staging_path: Path, metadata: dict, root: Path | None = None) -> Path:
    dest = canonical_path(metadata, root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != staging_path.resolve():
        shutil.copy2(staging_path, dest)
    return dest


def quarantine_file(src: Path, reason: str, root: Path | None = None) -> Path:
    dest = quarantine_dir(root) / f"{src.stem}_{abs(hash(reason)) % 10_000}{src.suffix}"
    shutil.copy2(src, dest)
    return dest
