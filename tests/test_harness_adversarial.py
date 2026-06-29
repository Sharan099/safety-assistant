from __future__ import annotations

import datetime
import json
import pytest
from pathlib import Path
from fastapi import HTTPException
from database.models import Test, TestResult, TestAuditLog, Chunk, Regulation, Document
from registry.harness_security import is_model_authorized, check_harness_access
from scripts.ingest_harness_test_records import validate_record_schema, ingest_records
from registry.harness_limits import extract_limit_details
from backend.app.gateway.gateway import LLMGateway


def test_adversarial_pass_fail_parsing():
    # Ceiling cases
    det1 = extract_limit_details("ThCC", "The Thorax Compression Criterion (ThCC) shall not exceed 42 mm;")
    assert det1 is not None
    val1, dir1, unit1 = det1
    assert val1 == 42.0
    assert dir1 == "ceiling"
    assert unit1 == "mm"

    det2 = extract_limit_details("HIC36", "The Head Injury Criterion (HIC36) must be less than 1000 g;")
    assert det2 is not None
    val2, dir2, unit2 = det2
    assert val2 == 1000.0
    assert dir2 == "ceiling"
    assert unit2 == "g"

    det3 = extract_limit_details("NIC", "NIC deflection is 15 ms maximum;")
    assert det3 is not None
    val3, dir3, unit3 = det3
    assert val3 == 15.0
    assert dir3 == "ceiling"
    assert unit3 == "ms"

    # Floor cases
    det4 = extract_limit_details("Force", "The seatbelt anchorage force must stay above 10 kN for safety;")
    assert det4 is not None
    val4, dir4, unit4 = det4
    assert val4 == 10.0
    assert dir4 == "floor"
    assert unit4 == "kn"

    # Ambiguous cases (should refuse)
    det5 = extract_limit_details("ThCC", "No clear compression limits are defined in this text.")
    assert det5 is None


def test_version_resolution_provenance_gaps(db_session, tmp_path):
    # Setup single regulation version (effective 2026-01-01)
    reg = Regulation(
        regulation_code="UN_R99",
        title="Reg 99",
        source_type="UNECE",
        amendment="Base",
        effective_date=datetime.date(2026, 1, 1),
        status="ACTIVE"
    )
    db_session.add(reg)
    db_session.commit()

    doc = Document(
        regulation_id=reg.id,
        document_name="UN_R99.pdf",
        document_type="PDF",
        hash="hash99",
        file_path="/storage/UN_R99.pdf"
    )
    db_session.add(doc)
    db_session.commit()

    chunk = Chunk(
        document_id=doc.id,
        chunk_text="Thorax Compression Criterion (ThCC) limit is 42 mm maximum.",
        chunk_index=1,
        page_number=5,
        section="5.3",
        chunk_type="clause"
    )
    db_session.add(chunk)
    db_session.commit()

    # 1. Date is before earliest version (2025 vs 2026) -> must resolve as REFUSED due to provenance gap
    record_early = {
        "test_id": "TEST-EARLY",
        "program": "P1",
        "date": "2025-06-28", # before earliest
        "test_type": "PHYSICAL_CRASH",
        "impact_mode": "FRONTAL_OFFSET",
        "results": [
            {
                "channel": "CH1",
                "injury_criterion": "ThCC",
                "value": 35.0,
                "linked_regulation_clause": "UN_R99#5.3"
            }
        ]
    }
    
    # Save mock file
    f_early = tmp_path / "early.json"
    with open(f_early, "w", encoding="utf-8") as fh:
        json.dump([record_early], fh)

    # Ingest using conftest db session by overriding ingest's SessionLocal
    import scripts.ingest_harness_test_records
    old_session = scripts.ingest_harness_test_records.SessionLocal
    scripts.ingest_harness_test_records.SessionLocal = lambda: db_session

    try:
        res = ingest_records(f_early)
        assert res["success"] == 0
        assert res["errors"] == 1
        assert res["outcomes"][0]["status"] == "REFUSED"
        assert "Provenance gap" in res["outcomes"][0]["reason"]
    finally:
        scripts.ingest_harness_test_records.SessionLocal = old_session


def test_missing_effective_date_refusal(db_session, tmp_path):
    # Setup regulation with null effective_date -> must refuse due to ambiguity
    reg = Regulation(
        regulation_code="UN_R99_AMB",
        title="Ambiguous Reg",
        source_type="UNECE",
        amendment="Base",
        effective_date=None,  # NULL!
        status="ACTIVE"
    )
    db_session.add(reg)
    db_session.commit()

    doc = Document(
        regulation_id=reg.id,
        document_name="UN_R99_AMB.pdf",
        document_type="PDF",
        hash="hash_amb",
        file_path="/storage/UN_R99_AMB.pdf"
    )
    db_session.add(doc)
    db_session.commit()

    chunk = Chunk(
        document_id=doc.id,
        chunk_text="Thorax Compression Criterion (ThCC) limit is 42 mm maximum.",
        chunk_index=1,
        page_number=5,
        section="5.3",
        chunk_type="clause"
    )
    db_session.add(chunk)
    db_session.commit()

    record = {
        "test_id": "TEST-AMB",
        "program": "P1",
        "date": "2026-06-28",
        "test_type": "PHYSICAL_CRASH",
        "impact_mode": "FRONTAL_OFFSET",
        "results": [
            {
                "channel": "CH1",
                "injury_criterion": "ThCC",
                "value": 35.0,
                "linked_regulation_clause": "UN_R99_AMB#5.3"
            }
        ]
    }
    f_amb = tmp_path / "amb.json"
    with open(f_amb, "w", encoding="utf-8") as fh:
        json.dump([record], fh)

    import scripts.ingest_harness_test_records
    old_session = scripts.ingest_harness_test_records.SessionLocal
    scripts.ingest_harness_test_records.SessionLocal = lambda: db_session

    try:
        res = ingest_records(f_amb)
        assert res["success"] == 0
        assert res["errors"] == 1
        assert res["outcomes"][0]["status"] == "REFUSED"
        assert "Ambiguous/missing effective_date" in res["outcomes"][0]["reason"]
    finally:
        scripts.ingest_harness_test_records.SessionLocal = old_session


def test_confidential_failover_gate_and_redaction():
    # Setup LLMGateway with mixed permitted/unpermitted chain
    # groq is permitted, openrouter is blocked/unpermitted
    gateway = LLMGateway(primary="groq")
    
    # Setup confidential context chunk
    confidential_chunks = [
        {
            "chunk_id": 1,
            "chunk_text": "Highly confidential dummy measurements showing ThCC peak 48.2 mm.",
            "confidential_tier": True
        }
    ]
    
    messages = [
        {"role": "user", "content": "Show our frontal test results."}
    ]

    # Force failover verification
    # We mock groq provider complete to raise a ProviderError, so it triggers failover to OpenRouter
    from backend.app.gateway.providers.base import ProviderError
    from backend.app.gateway.error_policy import ErrorKind
    
    def mock_groq_complete(*args, **kwargs):
        raise ProviderError("Groq service rate limited", kind=ErrorKind.RATE_LIMIT)
        
    gateway.groq.complete = mock_groq_complete

    # Since groq fails, chain falls over to openrouter (blocked).
    # Gateway must skip openrouter because of the confidential chunk, resulting in evidence-only.
    res = gateway.complete(messages, context_chunks=confidential_chunks)
    
    assert res.evidence_only is True
    
    # Assert openrouter was skipped
    router_step = next((s for s in res.steps if s.model_key == "openrouter_llama"), None)
    assert router_step is not None
    assert router_step.outcome == "skipped"
    assert router_step.detail == "unauthorized_confidential"

    # Assert evidence-only output has the chunk text redacted
    assert "[REDACTED:" in res.text
    assert "Highly confidential" not in res.text


def test_malformed_records_refusal(db_session, tmp_path):
    # Missing fields, incorrect criterion, and unit mismatch tests
    reg = Regulation(
        regulation_code="UN_R100",
        title="Reg 100",
        source_type="UNECE",
        amendment="Base",
        effective_date=datetime.date(2026, 1, 1),
        status="ACTIVE"
    )
    db_session.add(reg)
    db_session.commit()

    doc = Document(
        regulation_id=reg.id,
        document_name="UN_R100.pdf",
        document_type="PDF",
        hash="hash100",
        file_path="/storage/UN_R100.pdf"
    )
    db_session.add(doc)
    db_session.commit()

    chunk = Chunk(
        document_id=doc.id,
        chunk_text="Thorax Compression Criterion (ThCC) limit is 42 mm maximum.",
        chunk_index=1,
        page_number=5,
        section="5.3",
        chunk_type="clause"
    )
    db_session.add(chunk)
    db_session.commit()

    # Case A: Unit mismatch (record has 'g', limit is 'mm')
    rec_a = {
        "test_id": "TEST-UNIT-FAIL",
        "program": "P1",
        "date": "2026-06-28",
        "test_type": "PHYSICAL_CRASH",
        "impact_mode": "FRONTAL_OFFSET",
        "results": [
            {
                "channel": "CH1",
                "injury_criterion": "ThCC",
                "value": 35.0,
                "unit": "g", # Mismatch!
                "linked_regulation_clause": "UN_R100#5.3"
            }
        ]
    }
    
    # Case B: Criterion mismatch (record has 'HIC36', limit has 'ThCC')
    rec_b = {
        "test_id": "TEST-CRIT-FAIL",
        "program": "P1",
        "date": "2026-06-28",
        "test_type": "PHYSICAL_CRASH",
        "impact_mode": "FRONTAL_OFFSET",
        "results": [
            {
                "channel": "CH1",
                "injury_criterion": "HIC36", # Mismatch!
                "value": 500.0,
                "linked_regulation_clause": "UN_R100#5.3"
            }
        ]
    }

    import scripts.ingest_harness_test_records
    old_session = scripts.ingest_harness_test_records.SessionLocal
    scripts.ingest_harness_test_records.SessionLocal = lambda: db_session

    try:
        for idx, rec in enumerate([rec_a, rec_b]):
            f_path = tmp_path / f"malformed_{idx}.json"
            with open(f_path, "w", encoding="utf-8") as fh:
                json.dump([rec], fh)
            res = ingest_records(f_path)
            assert res["success"] == 0
            assert res["errors"] == 1
            assert res["outcomes"][0]["status"] == "REFUSED"
            
            # Verify quarantine files were created
            q_dir = Path(__file__).resolve().parents[1] / "data" / "quarantine_harness"
            assert q_dir.exists()
            assert len(list(q_dir.glob("refused_*.json"))) > 0
    finally:
        scripts.ingest_harness_test_records.SessionLocal = old_session
