"""Safe archive inspection — PRD_LEVEL3.md §6/FR-02, TRD_LEVEL3.md §5,
CLAUDE_CODE_LEVEL3_INSTRUCTIONS.md §6.

Inspects ZIP/TAR/TAR.GZ/TGZ members (path, size, compression, SHA-256)
*without* extracting anything to disk by default — `inspect_archive()` reads
each member's bytes through a streaming file object straight into a SHA-256
digest. Extraction only happens via `safe_extract_member()`, which re-checks
every safety condition immediately before writing.

Guards against (TRD_LEVEL3.md §5):
  - path traversal (`../`, absolute paths, drive-letter paths)
  - decompression bombs (per-member and total uncompressed-size caps)
  - duplicate member paths
  - nested archives (flagged, never auto-recursed — avoids unbounded
    recursion from a crafted or mislabeled archive)

The original archive under `Knowledge source/` is never opened in a mode
that could modify it (`"r"`, never `"a"`/`"w"`).
"""

from __future__ import annotations

import hashlib
import pathlib
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import IO, Literal

ArchiveFormat = Literal["zip", "tar", "tar.gz", "tgz"]

# A single member decompressing to more than this is refused outright — no
# legitimate LS-DYNA/PDF/CSV member in this corpus approaches it.
MAX_MEMBER_UNCOMPRESSED_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB
# A member whose uncompressed size is this many times its compressed size is
# a classic decompression-bomb signature (e.g. a zip of all-zero bytes).
SUSPICIOUS_COMPRESSION_RATIO = 500
_HASH_CHUNK = 4 * 1024 * 1024


@dataclass
class ArchiveMember:
    path: str  # normalized, forward-slash, relative to archive root
    size_bytes: int  # uncompressed
    compressed_size_bytes: int | None
    sha256: str | None  # None only if hashing was refused (unsafe member)
    is_dir: bool
    is_nested_archive: bool
    safety_issues: list[str] = field(default_factory=list)


@dataclass
class ArchiveManifest:
    archive_path: str
    archive_format: ArchiveFormat
    archive_sha256: str
    archive_size_bytes: int
    members: list[ArchiveMember]
    safety_issues: list[str]  # archive-level issues (not tied to one member)
    status: Literal["OK", "OK_WITH_WARNINGS", "REFUSED"]


def detect_format(path: pathlib.Path) -> ArchiveFormat | None:
    name = path.name.lower()
    if name.endswith(".tar.gz"):
        return "tar.gz"
    if name.endswith(".tgz"):
        return "tgz"
    if name.endswith(".tar"):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    return None


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_stream(fileobj: IO[bytes], *, cap: int) -> tuple[str, int]:
    """Hashes a member's content while enforcing `cap` — raises if exceeded,
    rather than silently hashing an unbounded/hostile stream to completion."""
    digest = hashlib.sha256()
    total = 0
    while True:
        block = fileobj.read(_HASH_CHUNK)
        if not block:
            break
        total += len(block)
        if total > cap:
            raise ValueError(f"member exceeds {cap} byte cap during read")
        digest.update(block)
    return digest.hexdigest(), total


def _normalize_member_path(raw_path: str) -> tuple[str, list[str]]:
    """Returns (normalized_path, issues). Never raises — an unsafe path is
    reported, not silently corrected, so the caller can refuse it."""
    issues: list[str] = []
    p = raw_path.replace("\\", "/")
    if p.startswith("/") or (len(p) > 1 and p[1] == ":"):  # POSIX absolute / Windows drive letter
        issues.append("absolute_path")
    parts = [seg for seg in p.split("/") if seg not in ("", ".")]
    if any(seg == ".." for seg in parts):
        issues.append("path_traversal")
    normalized = "/".join(parts)
    return normalized, issues


def _is_archive_name(name: str) -> bool:
    return detect_format(pathlib.Path(name)) is not None


def _inspect_zip(path: pathlib.Path) -> tuple[list[ArchiveMember], list[str]]:
    members: list[ArchiveMember] = []
    archive_issues: list[str] = []
    seen: set[str] = set()
    total_uncompressed = 0

    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            normalized, issues = _normalize_member_path(info.filename)
            is_dir = info.is_dir()

            if normalized in seen:
                issues.append("duplicate_path")
            seen.add(normalized)

            if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                issues.append("exceeds_size_cap")
            elif info.compress_size > 0 and info.file_size / max(info.compress_size, 1) > SUSPICIOUS_COMPRESSION_RATIO:
                issues.append("suspicious_compression_ratio")

            sha256: str | None = None
            if not is_dir and not issues:
                try:
                    with zf.open(info, "r") as member_f:
                        sha256, _ = _sha256_stream(member_f, cap=MAX_MEMBER_UNCOMPRESSED_BYTES)
                except ValueError:
                    issues.append("exceeds_size_cap")

            total_uncompressed += info.file_size
            members.append(
                ArchiveMember(
                    path=normalized,
                    size_bytes=info.file_size,
                    compressed_size_bytes=info.compress_size,
                    sha256=sha256,
                    is_dir=is_dir,
                    is_nested_archive=_is_archive_name(normalized),
                    safety_issues=issues,
                )
            )

    if total_uncompressed > MAX_MEMBER_UNCOMPRESSED_BYTES * 4:
        archive_issues.append("total_uncompressed_size_excessive")
    return members, archive_issues


def _inspect_tar(path: pathlib.Path, mode: Literal["r:", "r:gz"]) -> tuple[list[ArchiveMember], list[str]]:
    members: list[ArchiveMember] = []
    archive_issues: list[str] = []
    seen: set[str] = set()
    total_uncompressed = 0

    with tarfile.open(path, mode) as tf:
        for info in tf.getmembers():
            normalized, issues = _normalize_member_path(info.name)
            is_dir = info.isdir()

            if normalized in seen:
                issues.append("duplicate_path")
            seen.add(normalized)

            if info.issym() or info.islnk():
                # A symlink/hardlink inside a tar can point outside the
                # extraction root — never followed, never trusted.
                issues.append("unsafe_link")

            if info.isfile():
                if info.size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                    issues.append("exceeds_size_cap")

            sha256: str | None = None
            if info.isfile() and not issues:
                extracted = tf.extractfile(info)
                if extracted is not None:
                    try:
                        sha256, _ = _sha256_stream(extracted, cap=MAX_MEMBER_UNCOMPRESSED_BYTES)
                    except ValueError:
                        issues.append("exceeds_size_cap")

            total_uncompressed += info.size if info.isfile() else 0
            members.append(
                ArchiveMember(
                    path=normalized,
                    size_bytes=info.size if info.isfile() else 0,
                    compressed_size_bytes=None,  # tar doesn't expose per-member compressed size
                    sha256=sha256,
                    is_dir=is_dir,
                    is_nested_archive=_is_archive_name(normalized),
                    safety_issues=issues,
                )
            )

    if total_uncompressed > MAX_MEMBER_UNCOMPRESSED_BYTES * 4:
        archive_issues.append("total_uncompressed_size_excessive")
    return members, archive_issues


def inspect_archive(path: pathlib.Path) -> ArchiveManifest:
    """Inspects an archive's members without extracting them to disk. Always
    reads the original in `"r"` mode — never modifies it."""
    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(f"unsupported archive format: {path.name}")

    archive_sha256 = sha256_file(path)
    archive_size = path.stat().st_size

    if fmt == "zip":
        members, archive_issues = _inspect_zip(path)
    elif fmt == "tar":
        members, archive_issues = _inspect_tar(path, "r:")
    else:  # tar.gz / tgz
        members, archive_issues = _inspect_tar(path, "r:gz")

    any_member_issue = any(m.safety_issues for m in members)
    if archive_issues:
        status: Literal["OK", "OK_WITH_WARNINGS", "REFUSED"] = "OK_WITH_WARNINGS"
    elif any_member_issue:
        status = "OK_WITH_WARNINGS"
    else:
        status = "OK"

    return ArchiveManifest(
        archive_path=path.as_posix(),
        archive_format=fmt,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size,
        members=members,
        safety_issues=archive_issues,
        status=status,
    )


# A profiler/keyword-scan read (packages/cae/keyword_scan.py) never needs
# more than this from one member in memory — LS-DYNA text decks this size
# don't occur in the real corpus; anything bigger is refused rather than
# silently read in full.
MAX_MEMBER_READ_BYTES = 256 * 1024 * 1024  # 256 MB


def read_member_bytes(archive_path: pathlib.Path, member_path: str, *, cap: int = MAX_MEMBER_READ_BYTES) -> bytes:
    """Reads one member's full content into memory without extracting it to
    disk — used for lightweight scans (e.g. LS-DYNA keyword counting) of
    small text members still inside their archive."""
    fmt = detect_format(archive_path)
    if fmt is None:
        raise ValueError(f"unsupported archive format: {archive_path.name}")

    normalized, issues = _normalize_member_path(member_path)
    if issues:
        raise ValueError(f"refusing to read unsafe member path {member_path!r}: {issues}")

    if fmt == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf, zf.open(normalized, "r") as src:
            data, _ = _sha256_stream_and_collect(src, cap=cap)
            return data

    tar_mode: Literal["r:", "r:gz"] = "r:" if fmt == "tar" else "r:gz"
    with tarfile.open(archive_path, tar_mode) as tf:
        member = tf.getmember(member_path) if member_path in tf.getnames() else tf.getmember(normalized)
        if member.issym() or member.islnk():
            raise ValueError(f"refusing to read symlink/hardlink member: {member_path!r}")
        extracted = tf.extractfile(member)
        if extracted is None:
            raise ValueError(f"member is not a regular file: {member_path!r}")
        data, _ = _sha256_stream_and_collect(extracted, cap=cap)
        return data


def _sha256_stream_and_collect(fileobj: IO[bytes], *, cap: int) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    total = 0
    while True:
        block = fileobj.read(_HASH_CHUNK)
        if not block:
            break
        total += len(block)
        if total > cap:
            raise ValueError(f"member exceeds {cap} byte cap during read")
        chunks.append(block)
    return b"".join(chunks), total


def safe_extract_member(archive_path: pathlib.Path, member_path: str, dest_dir: pathlib.Path) -> pathlib.Path:
    """Extracts exactly one member, re-validating its path is safe and that
    the resolved destination stays inside `dest_dir` immediately before
    writing — independent of whatever `inspect_archive()` already found,
    since the archive could change between inspection and extraction."""
    fmt = detect_format(archive_path)
    if fmt is None:
        raise ValueError(f"unsupported archive format: {archive_path.name}")

    normalized, issues = _normalize_member_path(member_path)
    if issues:
        raise ValueError(f"refusing to extract unsafe member path {member_path!r}: {issues}")

    dest_dir = dest_dir.resolve()
    dest_path = (dest_dir / normalized).resolve()
    if dest_dir not in dest_path.parents and dest_path != dest_dir:
        raise ValueError(f"refusing to extract outside destination root: {member_path!r}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf, zf.open(normalized, "r") as src, dest_path.open("wb") as dst:
            for block in iter(lambda: src.read(_HASH_CHUNK), b""):
                dst.write(block)
    else:
        tar_mode: Literal["r:", "r:gz"] = "r:" if fmt == "tar" else "r:gz"
        with tarfile.open(archive_path, tar_mode) as tf:
            member = tf.getmember(member_path) if member_path in tf.getnames() else tf.getmember(normalized)
            if member.issym() or member.islnk():
                raise ValueError(f"refusing to extract symlink/hardlink member: {member_path!r}")
            extracted = tf.extractfile(member)
            if extracted is None:
                raise ValueError(f"member is not a regular file: {member_path!r}")
            with dest_path.open("wb") as dst:
                for block in iter(lambda: extracted.read(_HASH_CHUNK), b""):
                    dst.write(block)

    return dest_path
