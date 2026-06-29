import re
from typing import List, Dict, Any
from loguru import logger

class RegulationChunker:
    """
    Chunks parsed regulation pages into text blocks, preserving page number,
    detecting sections, and prepending hierarchical metadata to avoid cross-amendment confusion.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self, 
        pages: List[Dict[str, Any]], 
        regulation_metadata: Dict[str, Any],
        document_name: str
    ) -> List[Dict[str, Any]]:
        """
        Chunks a list of parsed pages and returns a list of chunk dicts ready for embedding.
        Each chunk dict has: chunk_text, chunk_index, page_number, section, paragraph.
        """
        chunks = []
        chunk_idx = 0
        
        # Build context header components
        reg_code = regulation_metadata.get("regulation_code", "UNKNOWN")
        amendment = regulation_metadata.get("amendment") or "Base"
        source_type = regulation_metadata.get("source_type", "INTERNAL")
        
        for page in pages:
            page_num = page["page_number"]
            text = page["text"]
            
            # Simple heuristic to find sections on this page (e.g., "5.2.1." or "Annex 3")
            sections_on_page = re.findall(r"\b((?:\d+\.)+\d+)\b", text)
            annexes_on_page = re.findall(r"\b(Annex\s+\d+)\b", text, re.IGNORECASE)
            
            section_ref = "General"
            if annexes_on_page:
                section_ref = annexes_on_page[0].title()
            elif sections_on_page:
                section_ref = sections_on_page[0]

            # Split text into overlapping blocks
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_payload = text[start:end]
                
                # Prepend the contextual metadata block to the chunk text
                # This guarantees that regulation-specific and series-specific info is embedded
                context_header = (
                    f"[Source: {source_type} | Reg: {reg_code} | Amendment: {amendment} | "
                    f"Doc: {document_name} | Page: {page_num} | Section: {section_ref}]\n\n"
                )
                
                final_chunk_text = context_header + chunk_payload
                
                chunks.append({
                    "chunk_text": final_chunk_text,
                    "chunk_index": chunk_idx,
                    "page_number": page_num,
                    "section": section_ref,
                    "paragraph": None  # Placeholder for paragraph-level parsing if needed
                })
                
                chunk_idx += 1
                start += (self.chunk_size - self.chunk_overlap)
                
                # Safety break to avoid infinite loop on bad inputs
                if self.chunk_size <= self.chunk_overlap:
                    break

        logger.info(f"Generated {len(chunks)} chunks for document {document_name}")
        return chunks
