import os
import fitz  # PyMuPDF
import pdfplumber
from loguru import logger
from typing import List, Dict, Any

class PDFParser:
    """
    Parser to extract text and tables from passive safety regulation PDF documents.
    Uses PyMuPDF for fast text extraction and pdfplumber for table extraction.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    def extract_metadata(self) -> Dict[str, Any]:
        """Extracts native PDF metadata."""
        metadata = {}
        try:
            with fitz.open(self.file_path) as doc:
                metadata = doc.metadata
                metadata["page_count"] = len(doc)
        except Exception as e:
            logger.error(f"Error extracting PDF metadata from {self.file_path}: {e}")
            metadata["page_count"] = 0
        return metadata

    def parse(self, extract_tables: bool = True) -> List[Dict[str, Any]]:
        """
        Parses the PDF page by page.
        Returns a list of dicts, each representing a page with its text, 
        markdown tables, and page number.
        """
        pages_data = []
        
        try:
            # We open the document with both fitz and pdfplumber
            with fitz.open(self.file_path) as doc:
                page_count = len(doc)
                
                # Open pdfplumber for table extraction
                plumber_pdf = None
                if extract_tables:
                    try:
                        plumber_pdf = pdfplumber.open(self.file_path)
                    except Exception as pe:
                        logger.warning(f"Could not open {self.file_path} with pdfplumber for tables. Falling back to PyMuPDF only: {pe}")

                for i in range(page_count):
                    page_num = i + 1
                    logger.debug(f"Parsing page {page_num}/{page_count} of {self.file_path}")
                    
                    # 1. Extract text with PyMuPDF
                    fitz_page = doc[i]
                    page_text = fitz_page.get_text()
                    
                    # 2. Extract tables with pdfplumber (if available)
                    table_mds = []
                    if plumber_pdf and i < len(plumber_pdf.pages):
                        try:
                            plumber_page = plumber_pdf.pages[i]
                            tables = plumber_page.extract_tables()
                            for table in tables:
                                if table:
                                    markdown_table = self._convert_table_to_markdown(table)
                                    if markdown_table:
                                        table_mds.append(markdown_table)
                        except Exception as te:
                            logger.error(f"Error extracting tables on page {page_num}: {te}")
                    
                    # 3. Combine text and table markdown
                    combined_text = page_text
                    if table_mds:
                        combined_text += "\n\n### Tables Extracted from Page:\n" + "\n\n".join(table_mds)
                    
                    pages_data.append({
                        "page_number": page_num,
                        "text": combined_text,
                        "raw_text": page_text,
                        "tables": table_mds
                    })
                    
                if plumber_pdf:
                    plumber_pdf.close()
                
        except Exception as e:
            logger.error(f"Fatal error parsing PDF {self.file_path}: {e}")
            raise e
            
        return pages_data

    def _convert_table_to_markdown(self, table: List[List[str]]) -> str:
        """Converts a list-of-lists table representation to GFM Markdown."""
        if not table or not table[0]:
            return ""
            
        markdown_lines = []
        
        # Clean values: remove None, replace newlines with spaces
        def clean_val(val: Any) -> str:
            if val is None:
                return ""
            return str(val).replace("\n", " ").replace("|", "\\|").strip()
            
        # Extract headers (first row)
        headers = [clean_val(h) for h in table[0]]
        markdown_lines.append("| " + " | ".join(headers) + " |")
        
        # Divider line
        markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Data rows
        for row in table[1:]:
            # Ensure row matches number of header columns
            cleaned_row = [clean_val(row[col_idx]) if col_idx < len(row) else "" for col_idx in range(len(headers))]
            markdown_lines.append("| " + " | ".join(cleaned_row) + " |")
            
        return "\n".join(markdown_lines)
