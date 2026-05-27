"""
PDF Linker - Creates clickable links to PDF documents with page numbers
"""
from pathlib import Path
from typing import Optional, Tuple
from config import REGULATIONS_DIR
import base64

def find_pdf_path(document_name: str) -> Optional[Path]:
    """
    Find the PDF file path by document name (stem)
    Searches in the regulations folder (recursive)
    """
    # Search in regulations folder (recursive)
    if REGULATIONS_DIR.exists():
        for pdf_path in REGULATIONS_DIR.rglob("*.pdf"):
            if pdf_path.stem.lower() == document_name.lower():
                return pdf_path
            # Also try partial matches
            if document_name.lower() in pdf_path.stem.lower():
                return pdf_path
    
    return None

def create_pdf_link(pdf_path: Path, page_number: int, section_number: Optional[str] = None) -> str:
    """
    Create a clickable link to PDF with page number
    Returns HTML link that opens PDF at specific page
    For Streamlit deployment, use the View PDF button instead
    """
    # Get relative path for display
    try:
        if pdf_path.is_relative_to(Path.cwd()):
            rel_path = pdf_path.relative_to(Path.cwd())
        else:
            rel_path = pdf_path.name
    except:
        rel_path = pdf_path.name
    
    # Create HTML link (file:// protocol for local, works in browser)
    file_url = pdf_path.as_uri()
    anchor = f"#page={page_number}"
    full_url = f"{file_url}{anchor}"
    
    # Create HTML link with better styling
    link_text = f"📄 {rel_path} - Page {page_number}"
    if section_number:
        link_text += f", Section {section_number}"
    
    return f'<a href="{full_url}" target="_blank" style="color: #1e88e5; text-decoration: none; font-weight: bold; padding: 0.3rem 0.6rem; background-color: #e3f2fd; border-radius: 0.3rem; display: inline-block; margin: 0.3rem 0;">{link_text} 🔗</a>'

def create_pdf_download_link(pdf_path: Path) -> str:
    """
    Create a download link for PDF (for deployment scenarios)
    """
    return f'<a href="/download_pdf?path={pdf_path.relative_to(Path.cwd())}" target="_blank" style="color: #1e88e5; text-decoration: none;">📥 Download PDF</a>'

def get_pdf_relative_path(pdf_path: Path) -> str:
    """Get relative path for serving PDFs in Streamlit"""
    try:
        # Try relative to REGULATIONS_DIR
        if pdf_path.is_relative_to(REGULATIONS_DIR):
            return str(pdf_path.relative_to(REGULATIONS_DIR))
        # Fallback to filename
        else:
            return pdf_path.name
    except:
        return pdf_path.name

