"""API smoke tests that don't require Qdrant/models loaded."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_health():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_pdf_missing_regulation():
    client = TestClient(app)
    res = client.get("/pdf/NOT-A-REG")
    assert res.status_code == 404


def test_citation_missing():
    client = TestClient(app)
    res = client.get("/citation/does-not-exist")
    assert res.status_code == 404
