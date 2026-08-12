from fastapi.testclient import TestClient

from apps.api.main import app
from tests.conftest import requires_db

client = TestClient(app)


@requires_db
def test_search_knowledge_over_http() -> None:
    resp = client.get("/api/v1/knowledge/search", params={"q": "frontal collision occupant protection", "limit": 5})
    assert resp.status_code == 200
    results = resp.json()
    assert results
    assert any(r["document_key"] == "unece-un-r94" for r in results)


@requires_db
def test_search_knowledge_filters_by_source_type() -> None:
    resp = client.get(
        "/api/v1/knowledge/search",
        params={"q": "material definition", "source_type": "OFFICIAL_DOCUMENTATION", "limit": 5},
    )
    assert resp.status_code == 200
    results = resp.json()
    assert all(r["source_type"] == "OFFICIAL_DOCUMENTATION" for r in results)
