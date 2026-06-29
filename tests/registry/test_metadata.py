import pytest
from datetime import date
from registry.metadata_extractor import MetadataExtractor

def test_parse_date():
    assert MetadataExtractor.parse_date("2025-01-01") == date(2025, 1, 1)
    assert MetadataExtractor.parse_date("23 June 2011") == date(2011, 6, 23)
    assert MetadataExtractor.parse_date("March 15, 2023") == date(2023, 3, 15)
    assert MetadataExtractor.parse_date("2026") == date(2026, 1, 1)
    assert MetadataExtractor.parse_date("invalid-date") is None

def test_extract_unece():
    extractor = MetadataExtractor()
    sample_text = (
        "AGREEMENT CONCERNING THE ADOPTION OF UNIFORM TECHNICAL PRESCRIPTIONS\n"
        "Regulation No. 95\n"
        "Uniform provisions concerning the approval of vehicles with regard to the protection\n"
        "of the occupants in the event of a lateral collision.\n"
        "Incorporating: 05 series of amendments - Date of entry into force: 2025-01-01\n"
    )
    meta = extractor.extract(sample_text, "r95_05_amendment.pdf")
    
    assert meta["source_type"] == "UNECE"
    assert meta["regulation_code"] == "R95"
    assert meta["amendment"] == "05 Series"
    assert meta["effective_date"] == date(2025, 1, 1)
    assert meta["market"] == "GLOBAL"
    assert "lateral collision" in meta["title"]

def test_extract_euroncap():
    extractor = MetadataExtractor()
    sample_text = (
        "EUROPEAN NEW CAR ASSESSMENT PROGRAMME (Euro NCAP)\n"
        "EURO NCAP Adult Occupant Protection (AOP) Assessment Protocol\n"
        "Version 9.5 - Implementation Year 2026\n"
        "Published November 2025\n"
    )
    meta = extractor.extract(sample_text, "euro_ncap_aop_2026.pdf")
    
    assert meta["source_type"] == "Euro NCAP"
    assert meta["regulation_code"] == "Euro NCAP 2026 - Adult Occupant Protection"
    assert meta["amendment"] == "Protocol Year 2026"
    assert meta["effective_date"] == date(2026, 1, 1)
    assert meta["market"] == "EU"

def test_extract_fmvss():
    extractor = MetadataExtractor()
    sample_text = (
        "DEPARTMENT OF TRANSPORTATION\n"
        "Standard No. 214 - Side Impact Protection\n"
        "This standard FMVSS 214 establishes performance requirements for side crashworthiness.\n"
    )
    meta = extractor.extract(sample_text, "fmvss_214.pdf")
    
    assert meta["source_type"] == "FMVSS"
    assert meta["regulation_code"] == "FMVSS 214"
    assert meta["market"] == "US"
    assert "Side Impact Protection" in meta["title"]

def test_extract_nhtsa_recall():
    extractor = MetadataExtractor()
    sample_text = (
        "NHTSA Recall Campaign Number 23V-123\n"
        "Subject: Frontal airbag module defect.\n"
        "Date issued: March 15, 2023\n"
    )
    meta = extractor.extract(sample_text, "nhtsa_recall_23v123.pdf")
    
    assert meta["source_type"] == "NHTSA"
    assert meta["regulation_code"] == "NHTSA Recall 23V-123"
    assert meta["effective_date"] == date(2023, 3, 15)
    assert meta["market"] == "US"

def test_extract_iihs():
    extractor = MetadataExtractor()
    sample_text = (
        "INSURANCE INSTITUTE FOR HIGHWAY SAFETY\n"
        "IIHS Side Impact Collision Test Procedure (Version II)\n"
        "Published June 2021\n"
    )
    meta = extractor.extract(sample_text, "iihs_side_test.pdf")
    
    assert meta["source_type"] == "IIHS"
    assert meta["regulation_code"] == "IIHS Side Impact"
    assert meta["effective_date"] == date(2021, 6, 1)
    assert meta["market"] == "US"
