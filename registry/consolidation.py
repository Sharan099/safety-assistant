import os
import json
import re
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from database.models import Regulation, Document, KnowledgeSection, Chunk

# Regexes for headers and clause numbers
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
CLAUSE_RE = re.compile(r"\b((?:\d+\.)+\d+|\d+|Annex\s+\d+|Appendix\s+[A-Z\d]+|Article\s+\d+)\b", re.IGNORECASE)

def split_markdown_into_sections(text: str) -> List[Dict[str, Any]]:
    """
    Parses a markdown string and splits it into logical sections by headers.
    Returns a list of dicts:
      {
         "section_number": str (e.g., "5.2.1" or "General"),
         "section_title": str,
         "content": str
      }
    """
    sections = []
    lines = text.split("\n")
    
    current_number = "General"
    current_title = "General Context"
    current_content = []
    
    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            # Save the current section before starting a new one
            if current_content:
                content_str = "\n".join(current_content).strip()
                if content_str:
                    sections.append({
                        "section_number": current_number,
                        "section_title": current_title,
                        "content": content_str
                    })
                current_content = []
            
            title = match.group(2).strip()
            # Try to extract a section number from title
            num_match = CLAUSE_RE.search(title)
            if num_match:
                current_number = num_match.group(1).strip()
                # Clean the number from the title
                current_title = title.replace(current_number, "").strip(".:- ")
                if not current_title:
                    current_title = f"Section {current_number}"
            else:
                current_number = "General"
                current_title = title
        else:
            current_content.append(line)
            
    # Append the last section
    if current_content:
        content_str = "\n".join(current_content).strip()
        if content_str:
            sections.append({
                "section_number": current_number,
                "section_title": current_title,
                "content": content_str
            })
        
    # Deduplicate / merge adjacent sections with same number if any
    merged_sections = []
    seen = {}
    for sec in sections:
        num = sec["section_number"]
        if num in seen:
            seen[num]["content"] += "\n\n" + sec["content"]
        else:
            seen[num] = sec
            merged_sections.append(sec)
            
    return merged_sections


class ConsolidationEngine:
    """
    Engine to consolidate base safety regulations with their subsequent 
    amendments, supplements, corrigenda, and technical bulletins.
    """
    
    @staticmethod
    def consolidate_regulation(db: Session, regulation_id: int) -> int:
        """
        Runs consolidation for a given regulation.
        1. Fetch all documents for this regulation.
        2. Sort documents by lineage: Base Regulation first, then Amendments/Supplements/Corrigenda.
        3. Drop existing knowledge_sections for this regulation.
        4. Reconstruct and write consolidated sections to knowledge_sections.
        
        Returns the number of consolidated sections created.
        """
        regulation = db.query(Regulation).filter(Regulation.id == regulation_id).first()
        if not regulation:
            return 0
            
        # Get all documents associated with the regulation
        docs = db.query(Document).filter(Document.regulation_id == regulation_id).all()
        if not docs:
            return 0
            
        # Sort documents: Base Regulation first, then Amendments, Supplements, Corrigenda, Technical Bulletins
        role_priority = {
            "BASE_REGULATION": 0,
            "AMENDMENT": 1,
            "SUPPLEMENT": 2,
            "CORRIGENDUM": 3,
            "TECHNICAL_BULLETIN": 4
        }
        
        # Sort helper
        def get_doc_sort_key(doc: Document) -> Tuple[int, int]:
            priority = role_priority.get(doc.document_role or "BASE_REGULATION", 99)
            # Within same role, sort by upload date or filename (as fallback)
            # Higher series/revisions are sorted properly by priority
            series_num = doc.series_number or 0
            rev_num = doc.revision_number or 0
            supp_num = doc.supplement_number or 0
            corr_num = doc.corrigendum_number or 0
            
            # Combine numbers to form a version ranking
            # e.g. series 5 > series 4
            version_score = (series_num * 1000) + (rev_num * 100) + (supp_num * 10) + corr_num
            return (priority, version_score)
            
        sorted_docs = sorted(docs, key=get_doc_sort_key)
        
        # Delete old knowledge sections for this regulation
        db.query(KnowledgeSection).filter(KnowledgeSection.regulation_id == regulation_id).delete()
        db.commit()
        
        consolidated_sections: Dict[str, Dict[str, Any]] = {}
        
        for doc in sorted_docs:
            # We need to read the document markdown content
            # Wait, our scanner writes markdown to output/markdown/<stem>.md!
            # Let's read the markdown file if it exists, or generate a fallback
            stem = os.path.splitext(doc.document_name)[0]
            md_path = os.path.join("output", "markdown", f"{stem}.md")
            
            markdown_text = ""
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown_text = f.read()
            else:
                # If markdown file does not exist, check if we can read the file_path PDF
                # and extract text as a fallback.
                if os.path.exists(doc.file_path):
                    from parser.pdf_parser import PDFParser
                    try:
                        parser = PDFParser(doc.file_path)
                        parsed = parser.parse()
                        markdown_text = "\n\n".join([p["text"] for p in parsed])
                    except Exception as exc:
                        markdown_text = f"# {doc.document_name}\n\nCould not extract content: {exc}"
                else:
                    markdown_text = f"# {doc.document_name}\n\nContent file not found."
            
            # Split this document's text into sections
            doc_sections = split_markdown_into_sections(markdown_text)
            
            # Apply sections to consolidated map
            is_base = (doc.document_role == "BASE_REGULATION")
            
            for sec in doc_sections:
                num = sec["section_number"]
                title = sec["section_title"]
                content = sec["content"]
                
                if not content.strip():
                    continue
                    
                if is_base:
                    # Insert base section
                    consolidated_sections[num] = {
                        "section_number": num,
                        "section_title": title,
                        "text_content": content,
                        "effective_document_id": doc.id,
                        "provenance_chain": [doc.document_name]
                    }
                else:
                    # It's an amendment/supplement/corrigendum
                    if num in consolidated_sections:
                        # Update existing section
                        orig = consolidated_sections[num]
                        # Merge content: append a note and the changes
                        orig["text_content"] = content # Override with the new amended text
                        orig["effective_document_id"] = doc.id
                        if doc.document_name not in orig["provenance_chain"]:
                            orig["provenance_chain"].append(doc.document_name)
                    else:
                        # New section introduced by the amendment
                        consolidated_sections[num] = {
                            "section_number": num,
                            "section_title": title,
                            "text_content": content,
                            "effective_document_id": doc.id,
                            "provenance_chain": [doc.document_name]
                        }
                        
        # Write consolidated sections to DB
        section_objs = []
        for num, sec_data in consolidated_sections.items():
            sec_obj = KnowledgeSection(
                regulation_id=regulation_id,
                section_number=sec_data["section_number"],
                section_title=sec_data["section_title"],
                text_content=sec_data["text_content"],
                effective_document_id=sec_data["effective_document_id"],
                provenance_chain=json.dumps(sec_data["provenance_chain"])
            )
            section_objs.append(sec_obj)
            
        db.bulk_save_objects(section_objs)
        db.commit()
        
        return len(section_objs)
