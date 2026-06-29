import pytest
from unittest.mock import patch, MagicMock
from registry.version_control import verify_and_register_document
from database.models import Regulation, Document

@pytest.fixture
def mock_db():
    return MagicMock()

@patch("registry.version_control.compute_file_hash")
def test_verify_and_register_new_regulation(mock_hash, mock_db):
    mock_hash.return_value = "checksum_new"
    
    # Mock database query: return None (no active regulation exists)
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    metadata = {
        "regulation_code": "R95",
        "title": "Lateral Collision Protection",
        "source_type": "UNECE",
        "amendment": "05 Series",
        "market": "GLOBAL"
    }
    
    reg, doc, status = verify_and_register_document(mock_db, "/mock/path/r95.pdf", metadata)
    
    assert status == "NEW"
    assert reg.regulation_code == "R95"
    assert reg.status == "ACTIVE"
    assert doc.document_name == "r95.pdf"
    assert doc.hash == "checksum_new"
    
    # Assert DB adds occurred
    assert mock_db.add.call_count == 2
    assert mock_db.commit.call_count == 2

@patch("registry.version_control.compute_file_hash")
def test_verify_and_register_no_change(mock_hash, mock_db):
    mock_hash.return_value = "checksum_same"
    
    # Mock database query: return an existing active regulation with the same checksum
    existing_reg = Regulation(
        id=42,
        regulation_code="R95",
        status="ACTIVE",
        checksum="checksum_same"
    )
    mock_db.query.return_value.filter.return_value.first.return_value = existing_reg
    
    # Mock document lookup
    existing_doc = Document(id=10, regulation_id=42, hash="checksum_same", document_name="r95.pdf")
    mock_db.query.return_value.filter.return_value.first.side_effect = [existing_reg, existing_doc]
    
    metadata = {
        "regulation_code": "R95",
        "source_type": "UNECE"
    }
    
    reg, doc, status = verify_and_register_document(mock_db, "/mock/path/r95.pdf", metadata)
    
    assert status == "NO_CHANGE"
    assert reg.id == 42
    assert doc.id == 10
    # No new adds/commits should be triggered for unchanged contents
    assert mock_db.commit.call_count == 0

@patch("registry.version_control.compute_file_hash")
def test_verify_and_register_superseded(mock_hash, mock_db):
    mock_hash.return_value = "checksum_different"
    
    # Mock database query: return an existing active regulation with a different checksum
    existing_reg = Regulation(
        id=42,
        regulation_code="R95",
        status="ACTIVE",
        checksum="checksum_old"
    )
    mock_db.query.return_value.filter.return_value.first.return_value = existing_reg
    
    metadata = {
        "regulation_code": "R95",
        "title": "Lateral Collision Protection",
        "source_type": "UNECE",
        "amendment": "05 Series"
    }
    
    reg, doc, status = verify_and_register_document(mock_db, "/mock/path/r95.pdf", metadata)
    
    assert status == "SUPERSEDED_UPDATED"
    # Ensure existing active is set to superseded
    assert existing_reg.status == "SUPERSEDED"
    # Ensure new active reg is created
    assert reg.status == "ACTIVE"
    assert doc.hash == "checksum_different"
    # Commit changes to DB
    assert mock_db.commit.call_count == 3
