import pytest
import json
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Regulation, Document, KnowledgeSection
from regulation_discovery.registry.version_parser import parse_document_metadata_from_filename
from registry.consolidation import split_markdown_into_sections, ConsolidationEngine

@pytest.fixture
def temp_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_filename_parser():
    # Base regulation series
    res = parse_document_metadata_from_filename("UN_R95_05Series.pdf")
    assert res["document_role"] == "BASE_REGULATION"
    assert res["is_complete_regulation"] is True
    assert res["series_number"] == 5
    assert res["regulation_code"] == "UN_R95"

    # Amendment
    res2 = parse_document_metadata_from_filename("EU_2021_1341_GSR_Amendment.pdf")
    assert res2["document_role"] == "AMENDMENT"
    assert res2["is_complete_regulation"] is False
    assert res2["regulation_code"] == "EU GSR2"

    # Technical Bulletin
    res3 = parse_document_metadata_from_filename("EuroNCAP_TB_THOR_Dummy.pdf")
    assert res3["document_role"] == "TECHNICAL_BULLETIN"
    assert res3["is_complete_regulation"] is False
    assert res3["regulation_code"] == "Euro NCAP THOR Dummy"

    # UNECE official browser download name
    res4 = parse_document_metadata_from_filename("R016r6am8e.pdf")
    assert res4["series_number"] == 8
    assert res4["regulation_code"] == "UN_R016"

def test_split_markdown_sections():
    md_text = """
# 1. Scope
This is the scope section content.

# 5.2.1. Dynamic Performance
This is the dynamic test performance requirements.
It should be consolidated.

## Annex 3 Barrier Specifications
Details about AE-MDB barrier specifications.
"""
    sections = split_markdown_into_sections(md_text)
    assert len(sections) == 3
    
    # Verify section numbers and titles
    numbers = [s["section_number"] for s in sections]
    assert "1" in numbers
    assert "5.2.1" in numbers
    assert "Annex 3" in numbers

    # Verify content matching
    sec_5 = next(s for s in sections if s["section_number"] == "5.2.1")
    assert "dynamic test performance requirements" in sec_5["content"]
    assert "consolidated" in sec_5["content"]

def test_consolidation_engine(temp_db, tmp_path):
    # Setup standard and documents in DB
    reg = Regulation(regulation_code="UN_R95", title="Lateral Collision Protection", source_type="UNECE")
    temp_db.add(reg)
    temp_db.commit()
    
    base_doc = Document(
        regulation_id=reg.id,
        document_name="UN_R95_05Series.pdf",
        document_type="PDF",
        file_path="/dummy/base.pdf",
        hash="hash1",
        document_role="BASE_REGULATION",
        is_complete_regulation=True,
        series_number=5
    )
    temp_db.add(base_doc)
    temp_db.commit()
    
    amend_doc = Document(
        regulation_id=reg.id,
        document_name="UN_R95_05Series_Amend1.pdf",
        document_type="PDF",
        file_path="/dummy/amend.pdf",
        hash="hash2",
        document_role="AMENDMENT",
        is_complete_regulation=False,
        series_number=5,
        parent_document_id=base_doc.id,
        applies_to_document_id=base_doc.id
    )
    temp_db.add(amend_doc)
    temp_db.commit()

    # Create dummy markdown outputs in temporary paths
    # Set output markdown folder mockup
    os.makedirs("output/markdown", exist_ok=True)
    
    base_md = """
# 1. Scope
Initial Scope.

# 5.2.1. Dynamic Performance
Initial dynamic limit is 42 mm.
"""
    amend_md = """
# 5.2.1. Dynamic Performance
Amended dynamic limit is reduced to 38 mm.
"""
    
    with open("output/markdown/UN_R95_05Series.md", "w", encoding="utf-8") as f:
        f.write(base_md)
    with open("output/markdown/UN_R95_05Series_Amend1.md", "w", encoding="utf-8") as f:
        f.write(amend_md)

    try:
        # Run consolidation
        count = ConsolidationEngine.consolidate_regulation(temp_db, reg.id)
        assert count == 2 # Section 1 and Section 5.2.1

        # Check section 1 (remains base)
        sec_1 = temp_db.query(KnowledgeSection).filter(
            KnowledgeSection.regulation_id == reg.id,
            KnowledgeSection.section_number == "1"
        ).first()
        assert sec_1 is not None
        assert sec_1.text_content == "Initial Scope."
        assert sec_1.effective_document_id == base_doc.id
        assert "UN_R95_05Series.pdf" in json.loads(sec_1.provenance_chain)

        # Check section 5.2.1 (overridden by amendment)
        sec_5 = temp_db.query(KnowledgeSection).filter(
            KnowledgeSection.regulation_id == reg.id,
            KnowledgeSection.section_number == "5.2.1"
        ).first()
        assert sec_5 is not None
        assert sec_5.text_content == "Amended dynamic limit is reduced to 38 mm."
        assert sec_5.effective_document_id == amend_doc.id
        chain = json.loads(sec_5.provenance_chain)
        assert "UN_R95_05Series.pdf" in chain
        assert "UN_R95_05Series_Amend1.pdf" in chain

    finally:
        # Cleanup mock files
        for fpath in ["output/markdown/UN_R95_05Series.md", "output/markdown/UN_R95_05Series_Amend1.md"]:
            if os.path.exists(fpath):
                os.remove(fpath)
