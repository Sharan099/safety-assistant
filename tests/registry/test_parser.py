import pytest
from unittest.mock import patch, MagicMock
from parser.pdf_parser import PDFParser

@patch("os.path.exists")
def test_parser_init_file_not_found(mock_exists):
    mock_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        PDFParser("nonexistent.pdf")

@patch("os.path.exists")
@patch("fitz.open")
@patch("pdfplumber.open")
def test_parser_extract_metadata(mock_plumber_open, mock_fitz_open, mock_exists):
    mock_exists.return_value = True
    
    # Mock PyMuPDF Document
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 5
    mock_doc.metadata = {"title": "Test Title"}
    mock_fitz_open.return_value.__enter__.return_value = mock_doc
    
    parser = PDFParser("test.pdf")
    meta = parser.extract_metadata()
    
    assert meta["page_count"] == 5
    assert meta["title"] == "Test Title"

@patch("os.path.exists")
@patch("fitz.open")
@patch("pdfplumber.open")
def test_parser_parse_pages(mock_plumber_open, mock_fitz_open, mock_exists):
    mock_exists.return_value = True
    
    # Mock PyMuPDF Document with 1 page
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page 1 Text Content"
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz_open.return_value.__enter__.return_value = mock_doc
    
    # Mock pdfplumber document and page
    mock_plumber_doc = MagicMock()
    mock_plumber_page = MagicMock()
    mock_plumber_page.extract_tables.return_value = [
        [["Col 1", "Col 2"], ["Row1 Val1", "Row1 Val2"]]
    ]
    mock_plumber_doc.pages = [mock_plumber_page]
    mock_plumber_open.return_value = mock_plumber_doc
    
    parser = PDFParser("test.pdf")
    pages_data = parser.parse()
    
    assert len(pages_data) == 1
    assert pages_data[0]["page_number"] == 1
    assert "Page 1 Text Content" in pages_data[0]["text"]
    assert "Col 1 | Col 2" in pages_data[0]["text"]
    assert "Row1 Val1 | Row1 Val2" in pages_data[0]["text"]
    assert len(pages_data[0]["tables"]) == 1
