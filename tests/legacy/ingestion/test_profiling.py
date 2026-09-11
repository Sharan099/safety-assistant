"""Source corpus profiler — PRD_LEVEL3.md §13, TRD_LEVEL3.md §6.

Builds a small crafted corpus in tmp_path mirroring the real "Knowledge
source/" shape (top-level PDFs, nested NHTSA-style family dirs, an archive
containing an LS-DYNA deck) rather than depending on the real 1.3 GB corpus
being present, so this test is fast and hermetic.
"""

import zipfile
from pathlib import Path, PurePosixPath

from packages.ingestion.profiling import (
    classify_file_type,
    classify_source_family,
    parser_candidate_for,
    profile_knowledge_sources,
)


def _build_corpus(root: Path) -> None:
    (root / "UN_R94.pdf").write_bytes(b"%PDF-1.4 fake\n")
    (root / "LS-DYNA_Manual_Theory_R17.pdf").write_bytes(b"%PDF-1.4 fake\n")

    family_dir = root / "NHTSA_vehicle_models" / "Honda_Accord_2014" / "ORIGINAL"
    family_dir.mkdir(parents=True)
    (family_dir / "report.pdf").write_bytes(b"%PDF-1.4 fake\n")

    zip_path = family_dir / "Oblique-Accord-Updated-PAB.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("main.key", "*KEYWORD\n*PART\nseat\n1,1,1\n*INCLUDE\nvehicle.k\n*END\n")
        zf.writestr("vehicle.k", "*NODE\n1,0.0,0.0,0.0\n*MAT_024\n1,7.85e-9\n")


def test_profile_discovers_top_level_files(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    rows = profile_knowledge_sources(tmp_path)
    top_level = {r.relative_path: r for r in rows if not r.archive_member}

    assert "UN_R94.pdf" in top_level
    assert top_level["UN_R94.pdf"].file_type == "pdf"
    assert top_level["UN_R94.pdf"].source_family == "unece_regulations"
    assert len(top_level["UN_R94.pdf"].sha256 or "") == 64

    assert "LS-DYNA_Manual_Theory_R17.pdf".lower() not in top_level  # case-sensitive path check below
    assert any(k.startswith("LS-DYNA_Manual") for k in top_level)


def test_profile_records_nested_family_and_archive_members(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    rows = profile_knowledge_sources(tmp_path)

    zip_row = next(r for r in rows if r.filename == "Oblique-Accord-Updated-PAB.zip")
    assert zip_row.source_family == "NHTSA_vehicle_models"
    assert zip_row.archive_member is False
    assert zip_row.parser_candidate == "archive_processor"

    member_rows = [r for r in rows if r.archive_member and r.archive_parent == zip_row.relative_path]
    assert {r.filename for r in member_rows} == {"main.key", "vehicle.k"}
    for r in member_rows:
        assert r.source_family == "NHTSA_vehicle_models"  # inherits the archive's family
        assert r.parser_candidate == "lsdyna_parser"
        assert r.sha256 is not None


def test_profile_scans_lsdyna_keywords_inside_archive_member(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    rows = profile_knowledge_sources(tmp_path)

    main_key = next(r for r in rows if r.archive_member and r.filename == "main.key")
    assert main_key.include_count == 1
    assert main_key.root_counts is not None
    assert main_key.root_counts.get("PART") == 1

    vehicle_k = next(r for r in rows if r.archive_member and r.filename == "vehicle.k")
    assert vehicle_k.root_counts is not None
    assert vehicle_k.root_counts.get("NODE") == 1
    assert vehicle_k.root_counts.get("MAT") == 1


def test_profile_scans_standalone_lsdyna_file(tmp_path: Path) -> None:
    (tmp_path / "standalone.k").write_text("*NODE\n1,0,0,0\n*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE\n", encoding="utf-8")
    rows = profile_knowledge_sources(tmp_path)
    row = next(r for r in rows if r.filename == "standalone.k")
    assert row.root_counts is not None
    assert row.root_counts.get("CONTACT") == 1


def test_classify_file_type_handles_tar_gz() -> None:
    assert classify_file_type(Path("Yaris.tar.gz")) == "tar.gz"
    assert classify_file_type(Path("neon-0.7.tar")) == "tar"
    assert classify_file_type(Path("main.key")) == "key"
    assert classify_file_type(Path("model.inc")) == "inc"


def test_classify_source_family_root_pdf_patterns() -> None:
    assert classify_source_family(PurePosixPath("UN_R94.pdf")) == "unece_regulations"
    assert classify_source_family(PurePosixPath("LS-DYNA_Users_Guide.pdf")) == "ls_dyna_docs"
    assert classify_source_family(PurePosixPath("cc-PAM-Crash-Spec-Sheet.pdf")) == "pam_crash_reference"
    assert classify_source_family(PurePosixPath("NHTSA_THOR/thor.pdf")) == "NHTSA_THOR"


def test_parser_candidate_for_csv_is_honestly_not_implemented() -> None:
    assert parser_candidate_for("csv") == "tabular_loader_NOT_IMPLEMENTED"
