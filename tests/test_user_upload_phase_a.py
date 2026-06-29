"""Phase A — structured confidential test-report upload."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from database.models import Chunk, Document, Regulation, Test, TestAuditLog, User, UserUpload
from registry.auth import hash_password
from registry.structured_test_pdf_parser import parse_structured_test_pdf
from scripts.generate_synthetic_test_pdf import build_pdf

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PDF = ROOT / "tests" / "fixtures" / "synthetic_test_report.pdf"


@pytest.fixture
def client(db_session, auth_user):
    from database.connection import get_db

    def _override():
        yield db_session

    app.dependency_overrides[get_db] = _override
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def auth_user(db_session):
    user = User(user_id="uploader", username="uploader", password_hash=hash_password("pass"))
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def r94_clause(db_session):
    reg = Regulation(
        regulation_code="UN_R94",
        title="UN R94",
        source_type="UNECE",
        amendment="05 Series",
        effective_date=datetime.date(2026, 1, 1),
        status="ACTIVE",
    )
    db_session.add(reg)
    db_session.commit()
    doc = Document(
        regulation_id=reg.id,
        document_name="UN_R94.pdf",
        document_type="PDF",
        hash="h1",
        file_path="/storage/UNECE/UN_R94.pdf",
    )
    db_session.add(doc)
    db_session.commit()
    chunk = Chunk(
        document_id=doc.id,
        chunk_text="The Thorax Compression Criterion (ThCC) shall not exceed 42 mm;",
        chunk_index=1,
        page_number=12,
        section="5.2.1.4",
        chunk_type="clause",
    )
    db_session.add(chunk)
    db_session.commit()
    return "UN_R94#5.2.1.4"


@pytest.fixture
def synthetic_pdf(tmp_path):
    with open(ROOT / "data" / "synthetic_test_report.json", encoding="utf-8") as fh:
        record = json.load(fh)[0]
    path = build_pdf(tmp_path / "synthetic.pdf", record)
    return path, record


def _login(client: TestClient) -> None:
    res = client.post("/api/v1/auth/login", json={"username": "uploader", "password": "pass"})
    assert res.status_code == 200


def test_parser_extracts_synthetic_pdf(synthetic_pdf, r94_clause):
    pdf_path, record = synthetic_pdf
    records, errors = parse_structured_test_pdf(pdf_path, linked_regulation_clause=r94_clause)
    assert not errors
    assert records[0]["test_id"] == record["test_id"]
    assert records[0]["results"][0]["injury_criterion"] == "ThCC"
    assert records[0]["results"][0]["value"] == 38.4


def test_user_upload_requires_auth(client, synthetic_pdf, r94_clause):
    pdf_path, _ = synthetic_pdf
    with open(pdf_path, "rb") as fh:
        res = client.post(
            "/api/v1/user-uploads",
            files={"file": ("report.pdf", fh, "application/pdf")},
            data={"linked_regulation_clause": r94_clause},
        )
    assert res.status_code == 401


def test_user_upload_happy_path(client, db_session, synthetic_pdf, r94_clause, auth_user):
    pdf_path, record = synthetic_pdf
    _login(client)
    with open(pdf_path, "rb") as fh:
        res = client.post(
            "/api/v1/user-uploads",
            files={"file": ("report.pdf", fh, "application/pdf")},
            data={"linked_regulation_clause": r94_clause},
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "ready"
    assert body["test_id"] == record["test_id"]

    test = db_session.query(Test).filter(Test.test_id == record["test_id"]).first()
    assert test is not None
    assert test.owner_user_id == "uploader"
    assert test.results[0].pass_fail in ("PASS", "FAIL")

    audit = db_session.query(TestAuditLog).filter(TestAuditLog.action == "UPLOAD").first()
    assert audit is not None
    assert audit.user_id == "uploader"

    upload_row = db_session.query(UserUpload).filter(UserUpload.id == body["upload_id"]).first()
    assert upload_row is not None
    assert "confidential" in upload_row.file_path.replace("\\", "/")


def test_cross_user_cannot_fetch_upload(client, db_session, synthetic_pdf, r94_clause, auth_user):
    pdf_path, _ = synthetic_pdf
    _login(client)
    with open(pdf_path, "rb") as fh:
        created = client.post(
            "/api/v1/user-uploads",
            files={"file": ("report.pdf", fh, "application/pdf")},
            data={"linked_regulation_clause": r94_clause},
        )
    upload_id = created.json()["upload_id"]

    other = User(user_id="other", username="other", password_hash=hash_password("x"))
    db_session.add(other)
    db_session.commit()
    client.post("/api/v1/auth/logout")
    client.post("/api/v1/auth/login", json={"username": "other", "password": "x"})
    res = client.get(f"/api/v1/user-uploads/{upload_id}")
    assert res.status_code == 404
