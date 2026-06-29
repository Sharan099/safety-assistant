import hashlib
import os
from loguru import logger
from sqlalchemy.orm import Session
from database.models import Regulation, Document

def compute_file_hash(file_path: str) -> str:
    """Computes the SHA-256 checksum of a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found for hashing: {file_path}")
        
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

def verify_and_register_document(
    db: Session, 
    file_path: str, 
    metadata: dict,
    source_url: str = None
) -> tuple[Regulation, Document, str]:
    """
    Registers a regulation and its source document in the database,
    verifying if the version already exists or if it supersedes an older one.
    
    Returns:
        (Regulation, Document, status_message)
        where status_message is one of: "NO_CHANGE", "SUPERSEDED_UPDATED", "NEW"
    """
    checksum = compute_file_hash(file_path)
    filename = os.path.basename(file_path)
    
    # Query for an existing ACTIVE regulation with the same code, amendment, supplement, corrigendum and source_type
    existing_active = db.query(Regulation).filter(
        Regulation.regulation_code == metadata["regulation_code"],
        Regulation.amendment == metadata.get("amendment"),
        Regulation.supplement == metadata.get("supplement"),
        Regulation.corrigendum == metadata.get("corrigendum"),
        Regulation.source_type == metadata["source_type"],
        Regulation.status == "ACTIVE"
    ).first()
    
    if existing_active:
        # Compare checksums
        if existing_active.checksum == checksum:
            logger.info(f"Regulation {metadata['regulation_code']} version matches existing active checksum. Skipping.")
            # Retrieve associated document
            doc = db.query(Document).filter(
                Document.regulation_id == existing_active.id,
                Document.hash == checksum
            ).first()
            if not doc:
                # If document entry is missing for some reason, recreate it
                doc = Document(
                    regulation_id=existing_active.id,
                    document_name=filename,
                    document_type="PDF",
                    file_path=file_path,
                    hash=checksum,
                    source_url=source_url or metadata.get("source_url")
                )
                db.add(doc)
                db.commit()
                db.refresh(doc)
            return existing_active, doc, "NO_CHANGE"
        else:
            logger.info(f"Checksum mismatch for regulation {metadata['regulation_code']}. Marking old version as SUPERSEDED.")
            # 1. Mark previous active version as SUPERSEDED
            existing_active.status = "SUPERSEDED"
            db.add(existing_active)
            db.commit()
            
            # 2. Create new active regulation entry
            new_reg = Regulation(
                regulation_code=metadata["regulation_code"],
                title=metadata["title"],
                source_type=metadata["source_type"],
                amendment=metadata.get("amendment"),
                revision=metadata.get("revision"),
                supplement=metadata.get("supplement"),
                corrigendum=metadata.get("corrigendum"),
                publication_date=metadata.get("publication_date"),
                effective_date=metadata.get("effective_date"),
                status="ACTIVE",
                market=metadata.get("market", "GLOBAL"),
                source_url=source_url or metadata.get("source_url"),
                checksum=checksum,
                local_file_path=file_path
            )
            db.add(new_reg)
            db.commit()
            db.refresh(new_reg)
            
            # Extract graph metadata
            from regulation_discovery.registry.version_parser import parse_document_metadata_from_filename
            meta_parsed = parse_document_metadata_from_filename(filename)
            
            # 3. Create document entry
            new_doc = Document(
                regulation_id=new_reg.id,
                document_name=filename,
                document_type="PDF",
                file_path=file_path,
                hash=checksum,
                source_url=source_url or metadata.get("source_url"),
                document_role=meta_parsed["document_role"],
                is_complete_regulation=meta_parsed["is_complete_regulation"],
                revision_number=meta_parsed["revision_number"],
                series_number=meta_parsed["series_number"],
                supplement_number=meta_parsed["supplement_number"],
                corrigendum_number=meta_parsed["corrigendum_number"]
            )
            # Find and link parent base document if this is an amendment/supplement/etc.
            if not new_doc.is_complete_regulation:
                parent_doc = db.query(Document).filter(
                    Document.regulation_id == new_reg.id,
                    Document.is_complete_regulation == True
                ).first()
                if parent_doc:
                    new_doc.parent_document_id = parent_doc.id
                    new_doc.applies_to_document_id = parent_doc.id

            db.add(new_doc)
            db.commit()
            db.refresh(new_doc)
            
            return new_reg, new_doc, "SUPERSEDED_UPDATED"
            
    else:
        # Case: Brand new regulation
        logger.info(f"Registering new regulation: {metadata['regulation_code']} ({metadata.get('amendment')})")
        
        new_reg = Regulation(
            regulation_code=metadata["regulation_code"],
            title=metadata["title"],
            source_type=metadata["source_type"],
            amendment=metadata.get("amendment"),
            revision=metadata.get("revision"),
            supplement=metadata.get("supplement"),
            corrigendum=metadata.get("corrigendum"),
            publication_date=metadata.get("publication_date"),
            effective_date=metadata.get("effective_date"),
            status="ACTIVE",
            market=metadata.get("market", "GLOBAL"),
            source_url=source_url or metadata.get("source_url"),
            checksum=checksum,
            local_file_path=file_path
        )
        db.add(new_reg)
        db.commit()
        db.refresh(new_reg)
        
        # Extract graph metadata
        from regulation_discovery.registry.version_parser import parse_document_metadata_from_filename
        meta_parsed = parse_document_metadata_from_filename(filename)
        
        new_doc = Document(
            regulation_id=new_reg.id,
            document_name=filename,
            document_type="PDF",
            file_path=file_path,
            hash=checksum,
            source_url=source_url or metadata.get("source_url"),
            document_role=meta_parsed["document_role"],
            is_complete_regulation=meta_parsed["is_complete_regulation"],
            revision_number=meta_parsed["revision_number"],
            series_number=meta_parsed["series_number"],
            supplement_number=meta_parsed["supplement_number"],
            corrigendum_number=meta_parsed["corrigendum_number"]
        )
        # Find and link parent base document if this is an amendment/supplement/etc.
        if not new_doc.is_complete_regulation:
            parent_doc = db.query(Document).filter(
                Document.regulation_id == new_reg.id,
                Document.is_complete_regulation == True
            ).first()
            if parent_doc:
                new_doc.parent_document_id = parent_doc.id
                new_doc.applies_to_document_id = parent_doc.id

        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        
        return new_reg, new_doc, "NEW"
