"""Document vocabulary for the product API (ADR-0029 §2): the internal `VersionStatus` state machine
is the truth; these are the user-facing stage names from 02_TRD/03_UI_UX and the public error table."""

from __future__ import annotations

from typing import Literal

SourceScope = Literal["AUTHORITATIVE_ORG", "WORKSPACE", "PRIVATE_USER"]
UPLOAD_SCOPES: tuple[str, ...] = ("PRIVATE_USER", "WORKSPACE")  # AUTHORITATIVE_ORG only via promotion
DOCUMENT_TYPES: tuple[str, ...] = ("REGULATION", "STANDARD", "TECHNICAL_REPORT", "MANUAL", "PROJECT_DOCUMENT")

DisplayStatus = Literal[
    "UPLOADED",
    "VALIDATING",
    "PARSING",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "VERIFYING",
    "READY",
    "FAILED",
    "QUARANTINED",
    "ARCHIVED",
]

_VERSION_TO_DISPLAY: dict[str, DisplayStatus] = {
    "DISCOVERED": "UPLOADED",
    "DOWNLOADED": "VALIDATING",
    "VALIDATED": "PARSING",
    "PARSED": "CHUNKING",
    "NORMALIZED": "CHUNKING",
    "CHUNKED": "EMBEDDING",
    "INDEXED": "VERIFYING",
    "VERIFIED": "VERIFYING",
    "ACTIVE": "READY",
    "SUPERSEDED": "ARCHIVED",
    "QUARANTINED": "QUARANTINED",
    "FAILED": "FAILED",
}

STAGES: tuple[DisplayStatus, ...] = (
    "UPLOADED",
    "VALIDATING",
    "PARSING",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "VERIFYING",
    "READY",
)


def display_status(version_status: str, *, job_status: str | None = None, archived: bool = False) -> DisplayStatus:
    """What the UI shows. A queued job on a fresh version is still UPLOADED; a version whose
    job is RUNNING shows the stage its status implies; terminal states pass through."""
    if archived:
        return "ARCHIVED"
    if job_status == "QUEUED" and version_status == "DISCOVERED":
        return "UPLOADED"
    return _VERSION_TO_DISPLAY.get(version_status, "FAILED")


# Public error messages (FR-DOC-05): safe wording keyed by code; the internal detail stays in
# ingestion_runs.error, referenced by ingestion_jobs.error_internal_ref.
PUBLIC_ERRORS: dict[str, str] = {
    "INVALID_PDF": "The file is not a valid PDF or failed integrity checks.",
    "TOO_LARGE": "The file exceeds the allowed size or page limit.",
    "UNREADABLE": "The PDF could not be read reliably (scanned pages without a text layer, or damaged pages).",
    "PROCESSING_ERROR": "Processing failed. Retry later or contact an administrator with the diagnostic reference.",
    "ATTEMPTS_EXHAUSTED": "Processing failed repeatedly. An administrator must review this document.",
}


def classify_error(run_status: str, detail: str | None) -> str:
    """Map a run outcome to a public error code without leaking internals."""
    d = (detail or "").lower()
    if run_status == "QUARANTINED":
        if "exceed" in d or "too large" in d or "page" in d and "limit" in d:
            return "TOO_LARGE"
        if "qa fail" in d or "failed page" in d or "text layer" in d or "scanned" in d:
            return "UNREADABLE"
        return "INVALID_PDF"
    return "PROCESSING_ERROR"
