"""Tests for coverage gap report (FR-17)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Regulation
from registry.coverage import build_coverage_report, load_expected_coverage


@pytest.fixture
def cov_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(
        Regulation(
            regulation_code="UN_R94",
            title="R94",
            source_type="UNECE",
            status="ACTIVE",
        )
    )
    session.add(
        Regulation(
            regulation_code="UN_R16",
            title="R16",
            source_type="UNECE",
            status="ACTIVE",
        )
    )
    session.commit()
    yield session
    session.close()


def test_expected_yaml_loads():
    cfg = load_expected_coverage()
    assert cfg["meta"]["region"] == "europe_only"
    assert "UNECE" in cfg["authorities"]


def test_coverage_report_shows_gaps(cov_db):
    report = build_coverage_report(cov_db)
    assert report["summary"]["expected"] > 0
    assert report["summary"]["ingested"] == 2
    assert report["summary"]["missing"] > 0
    assert "complete_count" in report["summary"]
    assert "partial_count" in report["summary"]
    assert "completeness_rate" in report["summary"]
    
    unece = next(a for a in report["authorities"] if a["authority"] == "UNECE")
    assert "UN_R94" not in unece["missing"]
    assert any(m.startswith("UN_R") for m in unece["missing"])
    assert "complete" in unece
    assert "partial" in unece
