"""Safe archive inspection — PRD_LEVEL3.md §6/FR-02, TRD_LEVEL3.md §5.

Builds real crafted archives (including malicious ones: zip-slip, absolute
paths, symlinks) in `tmp_path` rather than committing binary fixtures, so
each test is self-contained and the attack shape is visible in the test.
"""

import tarfile
import zipfile
from pathlib import Path
from typing import Literal

import pytest

from packages.ingestion.archives import inspect_archive, safe_extract_member


def _make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return path


def _make_tar(path: Path, entries: dict[str, bytes], *, gz: bool = False) -> Path:
    mode: Literal["w:gz", "w"] = "w:gz" if gz else "w"
    with tarfile.open(path, mode) as tf:
        for name, content in entries.items():
            import io

            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return path


def test_inspect_zip_records_members_with_sha256(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "deck.zip", {"main.key": b"*KEYWORD\n", "sub/vehicle.k": b"*NODE\n"})
    manifest = inspect_archive(archive)

    assert manifest.status == "OK"
    assert manifest.archive_format == "zip"
    assert len(manifest.archive_sha256) == 64
    paths = {m.path: m for m in manifest.members}
    assert "main.key" in paths
    assert "sub/vehicle.k" in paths
    assert paths["main.key"].sha256 is not None
    assert len(paths["main.key"].sha256) == 64
    assert paths["main.key"].safety_issues == []


def test_inspect_tar_gz_records_members(tmp_path: Path) -> None:
    archive = _make_tar(tmp_path / "model.tar.gz", {"model.k": b"*PART\n"}, gz=True)
    manifest = inspect_archive(archive)

    assert manifest.status == "OK"
    assert manifest.archive_format == "tar.gz"
    assert manifest.members[0].path == "model.k"
    assert manifest.members[0].sha256 is not None


def test_inspect_plain_tar(tmp_path: Path) -> None:
    archive = _make_tar(tmp_path / "model.tar", {"model.k": b"*PART\n"})
    manifest = inspect_archive(archive)
    assert manifest.archive_format == "tar"
    assert manifest.status == "OK"


def test_zip_slip_path_traversal_is_flagged_not_hashed(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "evil.zip", {"../../etc/passwd": b"pwned"})
    manifest = inspect_archive(archive)

    assert manifest.status == "OK_WITH_WARNINGS"
    member = manifest.members[0]
    assert "path_traversal" in member.safety_issues
    # Unsafe members are never hashed/opened — the issue is detected from
    # the path string alone, before any content read.
    assert member.sha256 is None


def test_absolute_path_member_is_flagged(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "abs.zip", {"/etc/passwd": b"pwned"})
    manifest = inspect_archive(archive)
    assert "absolute_path" in manifest.members[0].safety_issues


def test_duplicate_member_path_is_flagged(tmp_path: Path) -> None:
    # zipfile allows writing two entries with the same name.
    archive_path = tmp_path / "dup.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("a.k", b"*PART\n")
        zf.writestr("a.k", b"*NODE\n")
    manifest = inspect_archive(archive_path)
    assert manifest.status == "OK_WITH_WARNINGS"
    assert any("duplicate_path" in m.safety_issues for m in manifest.members)


def test_nested_archive_member_is_flagged_not_recursed(tmp_path: Path) -> None:
    inner = tmp_path / "inner.zip"
    _make_zip(inner, {"a.k": b"*PART\n"})
    outer = _make_zip(tmp_path / "outer.zip", {"inner.zip": inner.read_bytes()})
    manifest = inspect_archive(outer)
    assert manifest.members[0].is_nested_archive is True


def test_symlink_in_tar_is_flagged_and_never_followed(tmp_path: Path) -> None:
    archive_path = tmp_path / "link.tar"
    with tarfile.open(archive_path, "w") as tf:
        info = tarfile.TarInfo(name="evil_link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    manifest = inspect_archive(archive_path)
    assert "unsafe_link" in manifest.members[0].safety_issues
    assert manifest.members[0].sha256 is None


def test_safe_extract_member_writes_inside_dest_dir(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "deck.zip", {"main.key": b"*KEYWORD\n"})
    dest_dir = tmp_path / "extracted"
    out_path = safe_extract_member(archive, "main.key", dest_dir)
    assert out_path.read_bytes() == b"*KEYWORD\n"
    assert dest_dir.resolve() in out_path.resolve().parents


def test_safe_extract_member_refuses_path_traversal(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "evil.zip", {"../../etc/passwd": b"pwned"})
    dest_dir = tmp_path / "extracted"
    with pytest.raises(ValueError, match="unsafe member path"):
        safe_extract_member(archive, "../../etc/passwd", dest_dir)
    assert not dest_dir.exists()


def test_unsupported_extension_raises() -> None:
    with pytest.raises(ValueError, match="unsupported archive format"):
        inspect_archive(Path("something.rar"))
