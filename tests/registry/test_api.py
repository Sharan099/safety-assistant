import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from database.connection import get_db
from api.routes import get_search_engine

client = TestClient(app)

@pytest.fixture
def mock_db():
    db = MagicMock()
    # Configure a recursive mock query chain so any filter() call returns the query object
    query_mock = MagicMock()
    db.query.return_value = query_mock
    query_mock.filter.return_value = query_mock
    
    app.dependency_overrides[get_db] = lambda: db
    yield db
    app.dependency_overrides.pop(get_db, None)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data

def test_health_check_healthy(mock_db):
    # Mock database ping success
    mock_db.execute.return_value = True
    
    with patch("scheduler.celery_app.celery_app.control.ping") as mock_ping:
        mock_ping.return_value = {"worker1": "pong"}

        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "up"
        assert data["redis"] == "up"
        assert data["celery_worker"] == "up"

def test_list_regulations(mock_db):
    from database.models import Regulation
    mock_reg = Regulation(
        id=1,
        regulation_code="R95",
        title="Lateral Collision Protection",
        source_type="UNECE",
        status="ACTIVE"
    )
    
    # Configure the query mock to return our mock regulation
    mock_db.query.return_value.all.return_value = [mock_reg]
    
    response = client.get("/api/v1/regulations?regulation_code=R95&status=ACTIVE")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["regulation_code"] == "R95"

def test_upload_document():
    # Mock file upload
    file_content = b"%PDF-1.4 mock pdf content"
    files = {"file": ("test_doc.pdf", file_content, "application/pdf")}
    
    with patch("shutil.copyfileobj"), patch("os.path.getsize") as mock_size:
        mock_size.return_value = 1024
        response = client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "file_path" in data
        assert data["filename"] == "test_doc.pdf"
        assert data["size"] == 1024

@patch("scheduler.tasks.ingest_document_task.delay")
def test_ingest_document(mock_delay):
    mock_task = MagicMock()
    mock_task.id = "task_uuid_123"
    mock_delay.return_value = mock_task
    
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        
        payload = {
            "file_path": "/mock/uploads/r95.pdf",
            "manual_metadata": {"amendment": "05 Series"}
        }
        response = client.post("/api/v1/documents/ingest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_uuid_123"
        assert data["status"] == "queued"

def test_search_registry(mock_db):
    mock_engine = MagicMock()
    # Override get_search_engine dependency in app context
    app.dependency_overrides[get_search_engine] = lambda: mock_engine
    
    # Mock search response
    mock_engine.search.return_value = {
        "answer": "This is a grounded answer from source [1].",
        "sources": [
            {
                "chunk_id": 1,
                "chunk_text": "UN R95 lateral collision testing regulations specifications.",
                "page_number": 3,
                "regulation_code": "R95"
            }
        ],
        "metadata": {"latency_ms": 12.5}
    }
    
    payload = {
        "query": "What is the lateral collision barrier spec in R95?",
        "filter": {"regulation_code": "R95"}
    }
    try:
        response = client.post("/api/v1/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["regulation_code"] == "R95"
    finally:
        # Clean up dependency override
        app.dependency_overrides.pop(get_search_engine, None)
