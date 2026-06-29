"""Safe path construction for confidential upload storage."""

from __future__ import annotations

import re
from pathlib import Path

from app.config import settings

_SAFE_SEGMENT = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_CONFIDENTIAL_ROOT = (settings.ROOT_DIR / "storage" / "confidential").resolve()


def assert_safe_storage_segment(value: str, label: str) -> str:
    if not value or not _SAFE_SEGMENT.match(value):
        raise ValueError(f"Invalid {label} for storage path")
    return value


def confidential_upload_dir(user_id: str, upload_id: str) -> Path:
    safe_user = assert_safe_storage_segment(user_id, "user_id")
    safe_upload = assert_safe_storage_segment(upload_id, "upload_id")
    path = (settings.CONFIDENTIAL_UPLOAD_ROOT / safe_user / safe_upload).resolve()
    if not str(path).startswith(str(_CONFIDENTIAL_ROOT)):
        raise ValueError("Resolved upload path escapes confidential root")
    return path


def reject_confidential_ingest_path(path: Path | str) -> None:
    """Block regulation batch-ingest from scanning confidential storage."""
    resolved = Path(path).resolve()
    if str(resolved).startswith(str(_CONFIDENTIAL_ROOT)):
        raise ValueError(
            f"Refusing to ingest from confidential storage path: {resolved}. "
            "Use data/staging/ for public regulation PDFs only."
        )
