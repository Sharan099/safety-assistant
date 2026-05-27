"""
Document Processor for Safety Standards (ISO 26262, OEM manuals, etc.)
Extracts text with page numbers and section references
"""
import PyPDF2
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
import re
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with rich metadata"""
    text: str
    document_name: str
    page_number: int
    section_number: Optional[str] = None
    chunk_id: Optional[str] = None
    # Rich metadata tags
    origin: Optional[str] = None  # UNECE, NHTSA, Industry, OEM, etc.
    domain: Optional[str] = None  # Cybersecurity, Functional Safety, ADAS, Validation, Passive Safety, etc.
    strictness: Optional[str] = None  # Regulatory, Guideline, Standard, Best Practice
    method: Optional[str] = None  # ISO 26262, R155, R156, R94, R137, FMVSS 208, etc.
    year: Optional[int] = None
    source_type: Optional[str] = None  # Regulation, Guideline, Whitepaper, Protocol, etc.
    # Passive Safety specific metadata
    test_type: Optional[str] = None  # Frontal, Side, Pole, Pedestrian, Post-Crash
    metric: Optional[str] = None  # HIC, Chest_Deflection, Tibia_Index, Intrusion
    dummy_type: Optional[str] = None  # Hybrid-III, WorldSID, THOR-M
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "section_number": self.section_number,
            "chunk_id": self.chunk_id,
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
    """Processes PDF documents and extracts structured chunks with rich metadata"""
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        # Professional chunking: 500-700 tokens ≈ 375-525 words ≈ 600 chars
        self.chunk_size = chunk_size  # ~600 chars ≈ 500 tokens
        self.chunk_overlap = chunk_overlap  # ~100 chars ≈ 100 tokens
    
    def extract_metadata_from_path(self, pdf_path: Path) -> Dict[str, any]:
        """
        Extract metadata tags from file path and name
        Infers: Origin, Domain, Strictness, Method, Year
        """
        metadata = {
            "origin": None,
            "domain": None,
            "strictness": None,
            "method": None,
            "year": None,
            "source_type": None
        }
        
        # Get parent folder name
        parent_folder = pdf_path.parent.name.lower()
        file_name = pdf_path.stem.lower()
        
        # Extract Origin from folder structure
        if "unece" in parent_folder:
            metadata["origin"] = "UNECE"
            metadata["strictness"] = "Regulatory"
            metadata["source_type"] = "Regulation"
        elif "nhtsa" in parent_folder:
            metadata["origin"] = "NHTSA"
            metadata["strictness"] = "Guideline"
            metadata["source_type"] = "Guideline"
        elif "functional_safety" in parent_folder or "tuv" in file_name or "dekra" in file_name:
            metadata["origin"] = "Industry"
            metadata["strictness"] = "Standard"
            metadata["source_type"] = "Standard"
        elif "validation" in parent_folder:
            metadata["origin"] = "Industry"
            metadata["strictness"] = "Best Practice"
            metadata["source_type"] = "Whitepaper"
        else:
            metadata["origin"] = "Industry"
            metadata["strictness"] = "Standard"
            metadata["source_type"] = "Document"
        
        # Extract Domain from folder structure
        if "cybersecurity" in parent_folder or "r155" in file_name:
            metadata["domain"] = "Cybersecurity"
        elif "software_update" in parent_folder or "r156" in file_name:
            metadata["domain"] = "Software Update"
        elif "functional_safety" in parent_folder or "asil" in file_name or "iso_26262" in file_name:
            metadata["domain"] = "Functional Safety"
        elif "adas" in parent_folder or "adas" in file_name:
            metadata["domain"] = "ADAS"
        elif "driver_monitoring" in file_name or "dms" in file_name:
            metadata["domain"] = "Driver Monitoring"
        elif "validation" in parent_folder:
            metadata["domain"] = "Validation"
        else:
            metadata["domain"] = "General Safety"
        
        # Extract Method/Standard from file name
        if "r155" in file_name:
            metadata["method"] = "UNECE R155"
        elif "r156" in file_name:
            metadata["method"] = "UNECE R156"
        elif "iso_26262" in file_name or "iso26262" in file_name:
            metadata["method"] = "ISO 26262"
        elif "asil" in file_name:
            metadata["method"] = "ISO 26262 ASIL"
        elif "hara" in file_name:
            metadata["method"] = "HARA (ISO 26262)"
        
        # Extract Year from file name (4-digit year)
        year_match = re.search(r'\b(19|20)\d{2}\b', file_name)
        if year_match:
            metadata["year"] = int(year_match.group())
        
        return metadata
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Extract text from PDF with page numbers
        Returns list of {page_number, text} dictionaries
        """
        pages = []
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append({
                            "page_number": page_num,
                            "text": text.strip()
                        })
        except Exception as e:
            print(f"⚠️  pdfplumber failed for {pdf_path.name}: {e}, trying PyPDF2")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, start=1):
                        text = page.extract_text()
                        if text and text.strip():
                            pages.append({
                                "page_number": page_num,
                                "text": text.strip()
                            })
            except Exception as e2:
                print(f"❌ Error processing {pdf_path.name}: {e2}")
                return []
        
        return pages
    
    def extract_section_numbers(self, text: str) -> Optional[str]:
        """
        Extract section numbers from text (e.g., "5.2.3", "Section 4.1", "Clause 6.2.1")
        """
        # Pattern for ISO-style section numbers (e.g., 5.2.3, 6.1.2.4)
        iso_pattern = r'\b\d+\.\d+(?:\.\d+)*(?:\s+[A-Z][a-z]+)?'
        
        # Pattern for "Section X.Y" or "Clause X.Y"
        section_pattern = r'(?:Section|Clause|Part)\s+(\d+\.\d+(?:\.\d+)*)'
        
        # Try section/clause pattern first
        match = re.search(section_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Try ISO pattern at start of line or after newline
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            match = re.search(r'^' + iso_pattern, line.strip())
            if match:
                return match.group(0).split()[0] if ' ' in match.group(0) else match.group(0)
        
        return None
    
    def chunk_text(self, text: str, page_number: int, document_name: str, metadata: Dict = None) -> List[DocumentChunk]:
        """Chunk text into smaller pieces with overlap, cleaning garbled text"""
        if metadata is None:
            metadata = {}
        
        if not text or not text.strip():
            return []
        
        # Clean text first - remove common PDF extraction artifacts
        import re
        # Remove non-printable characters but keep essential symbols
        cleaned_text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
        # Fix common OCR/PDF extraction issues
        cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)  # Fix merged words
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single
        # Remove excessive special characters that indicate garbled text
        cleaned_text = re.sub(r'([^\w\s\-.,;:()\[\]{}%°±×÷≤≥≠≈∞∑∏∫√αβγδεθλμπστφω])\1{2,}', ' ', cleaned_text)
        
        # Use cleaned text for chunking
        text = cleaned_text
        
        # Split text into chunks with overlap (500-700 tokens, 100 token overlap)
        # Preserves page numbers and applies metadata tags
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        # Calculate overlap in words (approximately)
        overlap_words_count = max(1, len(words) * self.chunk_overlap // self.chunk_size) if words else 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space
            
            if current_length >= self.chunk_size:
                chunk_text = ' '.join(current_chunk)
                section_number = self.extract_section_numbers(chunk_text)
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    document_name=document_name,
                    page_number=page_number,
                    section_number=section_number,
                    chunk_id=f"{document_name}_p{page_number}_c{chunk_index}",
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
                
                # Keep overlap words for next chunk (100 token overlap)
                overlap_words = current_chunk[-overlap_words_count:] if len(current_chunk) > overlap_words_count else current_chunk
                current_chunk = overlap_words
                current_length = sum(len(w) + 1 for w in overlap_words)
                chunk_index += 1
        
        # Add remaining text as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            section_number = self.extract_section_numbers(chunk_text)
            # Check if chunk is readable (at least 60% alphanumeric)
            alnum_ratio = sum(1 for c in chunk_text if c.isalnum() or c.isspace()) / len(chunk_text) if chunk_text else 0
            if alnum_ratio >= 0.6:  # At least 60% readable
                chunk = DocumentChunk(
                    text=chunk_text,
                    document_name=document_name,
                    page_number=page_number,
                    section_number=section_number,
                    chunk_id=f"{document_name}_p{page_number}_c{chunk_index}",
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
        """
        Process a single PDF document and return chunks with rich metadata
        """
        document_name = pdf_path.stem
        print(f"📄 Processing {document_name}...")
        
        # Extract metadata from path
        metadata = self.extract_metadata_from_path(pdf_path)
        tags = [f"Origin={metadata['origin']}", f"Domain={metadata['domain']}", 
                f"Strictness={metadata['strictness']}"]
        if metadata.get('method'):
            tags.append(f"Method={metadata['method']}")
        if metadata.get('test_type'):
            tags.append(f"Test_Type={metadata['test_type']}")
        if metadata.get('metric'):
            tags.append(f"Metric={metadata['metric']}")
        if metadata.get('dummy_type'):
            tags.append(f"Dummy_Type={metadata['dummy_type']}")
        print(f"   Tags: {', '.join(tags)}")
        
        pages = self.extract_text_from_pdf(pdf_path)
        all_chunks = []
        
        for page_data in pages:
            page_chunks = self.chunk_text(
                page_data["text"],
                page_data["page_number"],
                document_name,
                metadata
            )
            all_chunks.extend(page_chunks)
        
        print(f"✅ Processed {document_name}: {len(pages)} pages, {len(all_chunks)} chunks")
        return all_chunks
    
    def process_directory(self, documents_dir: Path, recursive: bool = True) -> List[DocumentChunk]:
        """
        Process all PDFs in a directory (recursively by default)
        Supports the professional data structure:
        - data/unece_regulations/
        - data/nhtsa_guidelines/
        - data/functional_safety_concepts/
        - data/validation_testing/
        """
        if recursive:
            pdf_files = list(documents_dir.rglob("*.pdf"))
        else:
            pdf_files = list(documents_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {documents_dir}")
            return []
        
        print(f"📚 Found {len(pdf_files)} PDF file(s) to process")
        all_chunks = []
        for pdf_path in pdf_files:
            chunks = self.process_document(pdf_path)
            all_chunks.extend(chunks)
        
        return all_chunks

