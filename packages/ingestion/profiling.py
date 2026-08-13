"""Level-3 source corpus profiler — PRD_LEVEL3.md §13, TRD_LEVEL3.md §6,
CLAUDE_CODE_LEVEL3_INSTRUCTIONS.md §5.

Recursively discovers every file under the immutable `Knowledge source/`
root (see docs/ADR/0010 for the naming decision), records identity and
provenance for each — path, SHA-256, archive membership — and, for LS-DYNA
text members (standalone or inside an archive), a lightweight keyword scan
(`packages/cae/keyword_scan.py`; deliberately not full parsing here, per the
instructions above — that's `packages/cae/lsdyna/`).

Filesystem-only: no database, no network. Can run before
`docker compose up postgres`.
"""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Any

from packages.cae.keyword_scan import model_hints_for_path, scan_keywords
from packages.ingestion.archives import detect_format, inspect_archive, read_member_bytes, sha256_file

LSDYNA_EXTENSIONS = {".k", ".key", ".inc"}
MEMBER_ID_SEP = "::"


def classify_file_type(path: pathlib.PurePath) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith(".tar.gz"):
        return "tar.gz"
    if suffix == ".tgz":
        return "tgz"
    if suffix == ".tar":
        return "tar"
    if suffix == ".zip":
        return "zip"
    if suffix in LSDYNA_EXTENSIONS:
        return suffix.lstrip(".")
    if suffix in (".pdf", ".csv", ".sas", ".txt", ".md"):
        return suffix.lstrip(".")
    return suffix.lstrip(".") or "unknown"


def classify_source_family(relative_path: pathlib.PurePosixPath) -> str:
    """First path segment under the corpus root is the source family
    (e.g. "NHTSA_vehicle_models", "NHTSA_THOR", "OpenRadioss"). Files sitting
    directly at the root (the original V1 regulation/solver-manual PDFs) are
    grouped by filename pattern instead."""
    parts = relative_path.parts
    if len(parts) > 1:
        return parts[0]
    name = relative_path.name.upper()
    if name.startswith("UN_"):
        return "unece_regulations"
    if name.startswith("LS-DYNA") or name.startswith("LS_DYNA"):
        return "ls_dyna_docs"
    if "PAM" in name:
        return "pam_crash_reference"
    return "root"


def parser_candidate_for(file_type: str) -> str:
    if file_type in ("k", "key", "inc"):
        return "lsdyna_parser"
    if file_type == "pdf":
        return "pdf_pipeline"
    if file_type in ("zip", "tar", "tar.gz", "tgz"):
        return "archive_processor"
    if file_type in ("csv", "sas"):
        # No CSV/SAS loader exists yet in this repo. Recorded honestly as a
        # named-but-unimplemented candidate rather than silently "none".
        return "tabular_loader_NOT_IMPLEMENTED"
    return "none"


@dataclass
class SourceProfileRow:
    source_id: str
    relative_path: str
    filename: str
    extension: str
    size_bytes: int
    sha256: str | None
    archive_parent: str | None
    archive_member: bool
    file_type: str
    source_family: str
    parser_candidate: str
    status: str  # DISCOVERED | UNSAFE_SKIPPED
    keyword_counts: dict[str, int] | None = None
    include_count: int | None = None
    root_counts: dict[str, int] | None = None
    model_hints: list[str] = field(default_factory=list)
    safety_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _keyword_scan_row_fields(text: str, path_for_hints: str) -> dict[str, Any]:
    result = scan_keywords(text)
    return {
        "keyword_counts": result.keyword_counts,
        "include_count": result.include_count,
        "root_counts": result.root_counts,
        "model_hints": model_hints_for_path(path_for_hints),
    }


def _profile_top_level_file(root: pathlib.Path, path: pathlib.Path) -> SourceProfileRow:
    relative = pathlib.PurePosixPath(path.relative_to(root).as_posix())
    file_type = classify_file_type(path)
    row = SourceProfileRow(
        source_id=relative.as_posix(),
        relative_path=relative.as_posix(),
        filename=path.name,
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        archive_parent=None,
        archive_member=False,
        file_type=file_type,
        source_family=classify_source_family(relative),
        parser_candidate=parser_candidate_for(file_type),
        status="DISCOVERED",
    )
    if file_type in ("k", "key", "inc"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for key, value in _keyword_scan_row_fields(text, relative.as_posix()).items():
            setattr(row, key, value)
    return row


def _profile_archive(root: pathlib.Path, path: pathlib.Path) -> list[SourceProfileRow]:
    relative = pathlib.PurePosixPath(path.relative_to(root).as_posix())
    file_type = classify_file_type(path)
    rows: list[SourceProfileRow] = [
        SourceProfileRow(
            source_id=relative.as_posix(),
            relative_path=relative.as_posix(),
            filename=path.name,
            extension=path.suffix.lower(),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            archive_parent=None,
            archive_member=False,
            file_type=file_type,
            source_family=classify_source_family(relative),
            parser_candidate=parser_candidate_for(file_type),
            status="DISCOVERED",
        )
    ]

    manifest = inspect_archive(path)
    for member in manifest.members:
        if member.is_dir:
            continue
        member_path = pathlib.PurePosixPath(member.path)
        member_source_id = f"{relative.as_posix()}{MEMBER_ID_SEP}{member.path}"
        member_file_type = classify_file_type(member_path)
        member_row = SourceProfileRow(
            source_id=member_source_id,
            relative_path=member.path,
            filename=member_path.name,
            extension=member_path.suffix.lower(),
            size_bytes=member.size_bytes,
            sha256=member.sha256,
            archive_parent=relative.as_posix(),
            archive_member=True,
            file_type=member_file_type,
            source_family=classify_source_family(relative),  # inherits the archive's family
            parser_candidate=parser_candidate_for(member_file_type),
            status="UNSAFE_SKIPPED" if member.safety_issues else "DISCOVERED",
            safety_issues=list(member.safety_issues),
        )
        if member_file_type in ("k", "key", "inc") and not member.safety_issues:
            try:
                raw = read_member_bytes(path, member.path)
                text = raw.decode("utf-8", errors="replace")
                for key, value in _keyword_scan_row_fields(text, member.path).items():
                    setattr(member_row, key, value)
            except ValueError as exc:
                member_row.safety_issues.append(f"keyword_scan_skipped: {exc}")
        rows.append(member_row)
    return rows


def profile_knowledge_sources(root: pathlib.Path) -> list[SourceProfileRow]:
    """Walks `root` (the immutable `Knowledge source/` corpus) and returns one
    `SourceProfileRow` per top-level file plus one per safely-listable archive
    member. Never modifies anything under `root`."""
    rows: list[SourceProfileRow] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if detect_format(path) is not None:
            rows.extend(_profile_archive(root, path))
        else:
            rows.append(_profile_top_level_file(root, path))
    return rows
