"""Tests for offline staging batch ingest."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base
from registry.staging_manifest import gather_staging_pdfs, load_staging_manifest


@pytest.fixture
def staging_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    staging = tmp_path / "data" / "staging"
    staging.mkdir(parents=True)
    quarantine = tmp_path / "data" / "quarantine"
    quarantine.mkdir(parents=True)
    backups = tmp_path / "data" / "backups"
    backups.mkdir(parents=True)
    storage = tmp_path / "storage"
    storage.mkdir(parents=True)
    (tmp_path / "output").mkdir(parents=True)

    # Minimal coverage yaml
    coverage = tmp_path / "coverage_expected.yaml"
    coverage.write_text(
        """
authorities:
  UNECE:
    regulations:
      - { id: R16, title: "Safety-belts" }
      - { id: R94, title: "Frontal collision" }
""",
        encoding="utf-8",
    )

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield tmp_path, staging, session
    session.close()


def _make_pdf(path: Path, body: str) -> None:
    doc = fitz.open()
    filler = "Padding content for minimum file size requirement. " * 200
    for _ in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), body + "\n" + filler, fontsize=11)
    doc.save(path)
    doc.close()


def test_gather_staging_pdfs_skips_manifest(staging_env):
    _, staging, _ = staging_env
    _make_pdf(staging / "UN_R16_base.pdf", "UN REGULATION No. 16")
    (staging / "manifest.yaml").write_text("files: []\n", encoding="utf-8")
    (staging / "readme.txt").write_text("not a pdf", encoding="utf-8")

    pdfs = gather_staging_pdfs(staging)
    assert len(pdfs) == 1
    assert pdfs[0].name == "UN_R16_base.pdf"


def test_load_staging_manifest(staging_env):
    _, staging, _ = staging_env
    manifest = staging / "manifest.yaml"
    manifest.write_text(
        yaml.dump({
            "files": [
                {"filename": "messy.pdf", "regulation_code": "R16", "amendment": "Base"},
            ]
        }),
        encoding="utf-8",
    )
    mapping = load_staging_manifest(None, staging)
    assert "messy.pdf" in mapping
    assert mapping["messy.pdf"]["regulation_code"] == "UN_R16"
    assert mapping["messy.pdf"]["amendment"] == "Base"


def test_unmapped_file_quarantined_with_reason(staging_env):
    root, staging, session = staging_env
    bad = staging / "totally_unknown.pdf"
    _make_pdf(bad, "Some random document without regulation id.")

    import scripts.ingest_offline_reg as ingest_mod

    ingest_mod.ROOT = root
    with patch.object(ingest_mod, "load_expected_coverage") as load_cov:
        load_cov.return_value = {
            "authorities": {"UNECE": {"regulations": [{"id": "R16", "title": "R16"}]}}
        }
        entry = ingest_mod.process_file(bad, load_cov.return_value, session)

    assert entry["status"] == "UNMAPPED"
    assert "manifest.yaml" in entry.get("reason", "")
    assert not bad.exists()
    assert list((root / "data" / "quarantine").glob("*"))


def test_manifest_override_maps_messy_filename(staging_env):
    root, staging, session = staging_env
    messy = staging / "R016r6e.pdf"
    _make_pdf(messy, "UN REGULATION No. 16\nSafety-belts and restraint systems.")

    import scripts.ingest_offline_reg as ingest_mod

    ingest_mod.ROOT = root
    with (
        patch.object(ingest_mod, "load_expected_coverage") as load_cov,
        patch.object(ingest_mod, "ingest_document_task", return_value={"status": "success"}),
    ):
        load_cov.return_value = {
            "authorities": {"UNECE": {"regulations": [{"id": "R16", "title": "R16"}]}}
        }
        entry = ingest_mod.process_file(
            messy,
            load_cov.return_value,
            session,
            manual_reg_code="UN_R16",
            manual_amend="Base",
        )

    assert entry["status"] == "ACCEPTED"
    assert entry["reg_code"] == "UN_R16"


def test_invalid_pdf_rejected(staging_env):
    root, staging, session = staging_env
    bad = staging / "UN_R16_base.pdf"
    bad.write_bytes(b"not-a-pdf")

    import scripts.ingest_offline_reg as ingest_mod

    ingest_mod.ROOT = root
    with patch.object(ingest_mod, "load_expected_coverage") as load_cov:
        load_cov.return_value = {
            "authorities": {"UNECE": {"regulations": [{"id": "R16", "title": "R16"}]}}
        }
        entry = ingest_mod.process_file(bad, load_cov.return_value, session)

    assert entry["status"] == "REJECTED"
    assert "Magic bytes" in entry.get("reason", "")
