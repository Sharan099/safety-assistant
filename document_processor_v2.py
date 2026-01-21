"""
Enhanced Document Processor using PyMuPDF (fitz) for better PDF parsing
Extracts text with clause numbers, page numbers, and regulation metadata
"""
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
import re
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with regulation metadata"""
    text: str
    document_name: str
    page_number: int
    clause: Optional[str] = None  # Clause number (e.g., "5.2.1")
    section_number: Optional[str] = None
    chunk_id: Optional[str] = None
    regulation: Optional[str] = None  # Regulation name (e.g., "UNECE R94")
    source_pdf: Optional[str] = None  # PDF filename
    # Rich metadata tags
    origin: Optional[str] = None
    domain: Optional[str] = None
    strictness: Optional[str] = None
    method: Optional[str] = None
    year: Optional[int] = None
    source_type: Optional[str] = None
    test_type: Optional[str] = None
    metric: Optional[str] = None
    dummy_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "clause": self.clause,
            "section_number": self.section_number,
            "chunk_id": self.chunk_id,
            "regulation": self.regulation,
            "source_pdf": self.source_pdf,
            "origin": self.origin,
            "domain": self.domain,
            "strictness": self.strictness,
            "method": self.method,
            "year": self.year,
            "source_type": self.source_type,
            "test_type": self.test_type,
            "metric": self.metric,
            "dummy_type": self.dummy_type
        }

class DocumentProcessor:
    """Enhanced PDF processor using PyMuPDF with clause extraction"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size  # 300-600 tokens ≈ 500 chars
        self.chunk_overlap = chunk_overlap  # 50 tokens overlap
    
    def extract_regulation_name(self, pdf_path: Path) -> Optional[str]:
        """Extract regulation name from filename or path"""
        filename = pdf_path.stem.upper()
        
        # Pattern matching for common regulations
        if "R94" in filename or "R_94" in filename:
            return "UNECE R94"
        elif "R95" in filename or "R_95" in filename:
            return "UNECE R95"
        elif "R137" in filename or "R_137" in filename:
            return "UNECE R137"
        elif "R155" in filename or "R_155" in filename:
            return "UNECE R155"
        elif "R156" in filename or "R_156" in filename:
            return "UNECE R156"
        elif "FMVSS" in filename or "FMVSS_208" in filename:
            return "FMVSS 208"
        elif "EURNCAP" in filename or "EURO_NCAP" in filename:
            return "Euro NCAP"
        elif "ISO_26262" in filename or "ISO26262" in filename:
            return "ISO 26262"
        
        # Try to extract from path
        parent = pdf_path.parent.name.upper()
        if "UNECE" in parent:
            # Extract R number from filename
            r_match = re.search(r'R[_\s]?(\d+)', filename)
            if r_match:
                return f"UNECE R{r_match.group(1)}"
        elif "NHTSA" in parent or "FMVSS" in parent:
            fmvss_match = re.search(r'FMVSS[_\s]?(\d+)', filename)
            if fmvss_match:
                return f"FMVSS {fmvss_match.group(1)}"
        
        return None
    
    def extract_clause_number(self, text: str) -> Optional[str]:
        """
        Extract clause number from text
        Patterns: "5.2.1", "Section 3.4", "Clause 2.1.3", "§ 4.2"
        """
        # Pattern 1: Decimal notation (5.2.1, 3.4.2.1)
        clause_patterns = [
            r'\b(\d+\.\d+(?:\.\d+)*(?:\.\d+)?)\b',  # 5.2.1 or 3.4.2.1
            r'(?:Section|Clause|§|Article)\s+(\d+\.\d+(?:\.\d+)*)',  # Section 5.2.1
            r'(?:Annex|Appendix)\s+([A-Z]|\d+)',  # Annex A, Appendix 1
        ]
        
        for pattern in clause_patterns:
            match = re.search(pattern, text[:200])  # Check first 200 chars
            if match:
                return match.group(1)
        
        return None
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Extract text from PDF using PyMuPDF (fitz) with page numbers
        Falls back to pdfplumber if PyMuPDF not available
        """
        pages = []
        
        # Try PyMuPDF first (better layout preservation)
        if fitz:
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")  # Get text with layout
                    
                    if text and text.strip():
                        pages.append({
                            "page_number": page_num + 1,
                            "text": text.strip()
                        })
                doc.close()
                return pages
            except Exception as e:
                print(f"⚠️ PyMuPDF extraction failed: {e}, trying pdfplumber...")
        
        # Fallback to pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append({
                            "page_number": page_num,
                            "text": text.strip()
                        })
        except Exception as e:
            print(f"⚠️ pdfplumber extraction failed: {e}")
        
        return pages
    
    def hierarchical_chunk_text(self, text: str, page_number: int, document_name: str, 
                                regulation: Optional[str], metadata: Dict) -> List[DocumentChunk]:
        """
        Hierarchical chunking: Split by regulation → section → clause
        Chunk size: 300-600 tokens (500 chars), overlap: 50 tokens
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        
        # Clean text
        cleaned_text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
        cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Try to split by sections/clauses first (hierarchical)
        # Pattern: Section/Clause numbers (e.g., "5.2.1", "Section 3.4")
        section_pattern = r'(?:^|\n)\s*(?:Section|Clause|§|Article|Annex)\s+(\d+\.\d+(?:\.\d+)*)'
        section_matches = list(re.finditer(section_pattern, cleaned_text, re.MULTILINE | re.IGNORECASE))
        
        if section_matches and len(section_matches) > 1:
            # Split by sections
            for i, match in enumerate(section_matches):
                start_idx = match.start()
                end_idx = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(cleaned_text)
                section_text = cleaned_text[start_idx:end_idx].strip()
                clause_num = match.group(1)
                
                if section_text:
                    # Further chunk if section is too long
                    section_chunks = self._chunk_by_size(section_text, clause_num, page_number, 
                                                       document_name, regulation, metadata)
                    chunks.extend(section_chunks)
        else:
            # No clear sections, use size-based chunking
            chunks = self._chunk_by_size(cleaned_text, None, page_number, 
                                        document_name, regulation, metadata)
        
        return chunks
    
    def _chunk_by_size(self, text: str, clause: Optional[str], page_number: int,
                       document_name: str, regulation: Optional[str], metadata: Dict) -> List[DocumentChunk]:
        """Chunk text by size with overlap"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        # Calculate overlap in words
        overlap_words_count = max(1, len(words) * self.chunk_overlap // self.chunk_size) if words else 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= self.chunk_size:
                chunk_text = ' '.join(current_chunk)
                
                # Filter garbled chunks
                alphanumeric_chars = sum(c.isalnum() for c in chunk_text)
                if len(chunk_text) > 0 and (alphanumeric_chars / len(chunk_text)) < 0.60:
                    current_chunk = []
                    current_length = 0
                    continue
                
                # Extract clause if not provided
                if not clause:
                    clause = self.extract_clause_number(chunk_text)
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    document_name=document_name,
                    page_number=page_number,
                    clause=clause,
                    section_number=clause,  # Use clause as section for now
                    chunk_id=f"{document_name}_p{page_number}_c{chunk_index}",
                    regulation=regulation,
                    source_pdf=metadata.get("source_pdf"),
                    origin=metadata.get("origin"),
                    domain=metadata.get("domain"),
                    strictness=metadata.get("strictness"),
                    method=metadata.get("method"),
                    year=metadata.get("year"),
                    source_type=metadata.get("source_type"),
                    test_type=metadata.get("test_type"),
                    metric=metadata.get("metric"),
                    dummy_type=metadata.get("dummy_type")
                )
                chunks.append(chunk)
                
                # Keep overlap words
                overlap_words = current_chunk[-overlap_words_count:] if len(current_chunk) > overlap_words_count else current_chunk
                current_chunk = overlap_words
                current_length = sum(len(w) + 1 for w in overlap_words)
                chunk_index += 1
        
        # Add remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            alphanumeric_chars = sum(c.isalnum() for c in chunk_text)
            if len(chunk_text) > 0 and (alphanumeric_chars / len(chunk_text)) >= 0.60:
                if not clause:
                    clause = self.extract_clause_number(chunk_text)
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    document_name=document_name,
                    page_number=page_number,
                    clause=clause,
                    section_number=clause,
                    chunk_id=f"{document_name}_p{page_number}_c{chunk_index}",
                    regulation=regulation,
                    source_pdf=metadata.get("source_pdf"),
                    origin=metadata.get("origin"),
                    domain=metadata.get("domain"),
                    strictness=metadata.get("strictness"),
                    method=metadata.get("method"),
                    year=metadata.get("year"),
                    source_type=metadata.get("source_type"),
                    test_type=metadata.get("test_type"),
                    metric=metadata.get("metric"),
                    dummy_type=metadata.get("dummy_type")
                )
                chunks.append(chunk)
        
        return chunks
    
    def process_document(self, pdf_path: Path) -> List[DocumentChunk]:
        """Process a single PDF document"""
        document_name = pdf_path.stem
        print(f"📄 Processing {document_name}...")
        
        # Extract regulation name
        regulation = self.extract_regulation_name(pdf_path)
        
        # Extract document-level metadata
        from document_processor import DocumentProcessor as LegacyProcessor
        legacy_processor = LegacyProcessor()
        doc_metadata = legacy_processor.extract_metadata_from_path(pdf_path)
        doc_metadata["source_pdf"] = pdf_path.name
        doc_metadata["regulation"] = regulation
        
        # Extract pages
        pages = self.extract_text_from_pdf(pdf_path)
        all_chunks = []
        
        for page_data in pages:
            page_chunks = self.hierarchical_chunk_text(
                page_data["text"],
                page_data["page_number"],
                document_name,
                regulation,
                doc_metadata
            )
            all_chunks.extend(page_chunks)
        
        print(f"✅ Processed {document_name}: {len(pages)} pages, {len(all_chunks)} chunks")
        return all_chunks
    
    def process_directory(self, documents_dir: Path, recursive: bool = False) -> List[DocumentChunk]:
        """Process all PDFs in a directory"""
        pdf_files = []
        if recursive:
            pdf_files = list(documents_dir.rglob("*.pdf"))
        else:
            pdf_files = list(documents_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {documents_dir}")
            return []
        
        all_chunks = []
        for pdf_path in pdf_files:
            chunks = self.process_document(pdf_path)
            all_chunks.extend(chunks)
        
        return all_chunks


