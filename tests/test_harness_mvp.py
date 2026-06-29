from __future__ import annotations

import datetime
import pytest
from fastapi import HTTPException
from database.models import Test, TestResult, TestAuditLog, Chunk, Regulation, Document
from registry.harness_security import is_model_authorized, check_harness_access, audit_access
from scripts.ingest_harness_test_records import validate_record_schema
from registry.harness_limits import extract_limit_details, verify_criterion_matches_clause, derive_pass_fail, compute_headroom_mm
from registry.margin_query import (
    MARGIN_TRANSCRIPT_TURN_A,
    MARGIN_TRANSCRIPT_TURN_B,
    parse_measurement_pair,
    try_margin_query_response,
)


def test_validate_record_schema():
    # Valid normalized record
    valid = {
        "test_id": "TEST-1",
        "program": "P1",
        "date": "2026-06-28",
        "test_type": "PHYSICAL_CRASH",
        "impact_mode": "FRONTAL_OFFSET",
        "results": [
            {
                "channel": "CH1",
                "injury_criterion": "ThCC",
                "value": 38.4,
                "linked_regulation_clause": "UN_R94#5.2.1.4"
            }
        ]
    }
    assert len(validate_record_schema(valid)) == 0

    # Missing test fields
    invalid_test = {
        "test_id": "TEST-1",
        "program": "P1",
        "date": "2026-06-28"
    }
    assert len(validate_record_schema(invalid_test)) > 0


def test_is_model_authorized():
    # Local Ollama is default allowed/trusted
    assert is_model_authorized("ollama", "llama3") is True
    assert is_model_authorized("local_model", "custom") is True

    # Paid Groq / Anthropic allowed
    assert is_model_authorized("groq", "llama-3.3-70b-versatile") is True
    assert is_model_authorized("groq_power", "llama-3.3-70b-versatile") is True
    assert is_model_authorized("groq_fast", "openai/gpt-oss-20b") is True
    assert is_model_authorized("anthropic_sonnet", "claude-sonnet-4") is True

    # OpenRouter is blocked by default as "VERIFY-BEFORE-USE"
    assert is_model_authorized("openrouter_llama", "meta-llama/llama-3.3-70b-instruct") is False
    assert is_model_authorized("openrouter_claude", "anthropic/claude-sonnet-4") is False

    # Gemini & Free tiers blocked
    assert is_model_authorized("openrouter_gemini", "google/gemini-2.5-flash") is False
    assert is_model_authorized("free_gemini", "gemini-flash") is False
    assert is_model_authorized("groq_free", "llama-free") is False


def test_harness_security_checks_and_audit(db_session):
    # Setup test event in DB
    t = Test(
        test_id="TEST-SEC-001",
        program="SUV-EV",
        date=datetime.date(2026, 6, 28),
        test_type="PHYSICAL_CRASH",
        impact_mode="FRONTAL_OFFSET",
        dummy="Hybrid III",
        confidential_tier=True,
        owner_user_id="eng-1",
    )
    db_session.add(t)
    db_session.commit()

    # Access via authorized model (Ollama)
    check_harness_access(db_session, "ollama", "llama3", "eng-1", ["TEST-SEC-001"])
    
    # Audit log check
    audit = db_session.query(TestAuditLog).filter(TestAuditLog.resource == "TEST-SEC-001").first()
    assert audit is not None
    assert audit.user_id == "eng-1"
    assert "AUTHORIZED" in audit.details

    # Access via unauthorized model (same owner — model gate, not ownership)
    with pytest.raises(HTTPException) as excinfo:
        check_harness_access(db_session, "openrouter_llama", "meta-llama/llama-3.3-70b-instruct", "eng-1", ["TEST-SEC-001"])
    
    assert excinfo.value.status_code == 403
    
    # Check failed audit entry exists (model unauthorized)
    failed_audit = db_session.query(TestAuditLog).filter(
        TestAuditLog.user_id == "eng-1",
        TestAuditLog.resource == "TEST-SEC-001",
    ).order_by(TestAuditLog.id.desc()).first()
    assert failed_audit is not None
    assert "UNAUTHORIZED" in failed_audit.details

    # Different user cannot access another owner's confidential test
    with pytest.raises(HTTPException) as owner_exc:
        check_harness_access(db_session, "ollama", "llama3", "eng-2", ["TEST-SEC-001"])
    assert owner_exc.value.status_code == 403
    owner_denied = db_session.query(TestAuditLog).filter(
        TestAuditLog.user_id == "eng-2",
        TestAuditLog.resource == "TEST-SEC-001",
    ).first()
    assert owner_denied is not None
    assert "OWNER_DENIED" in owner_denied.details


def test_linking_validation_and_derived_pass_fail(db_session):
    # 1. Setup two versions of UN R94 regulation
    # Version 1: 04 Series (effective 2021) with 42mm chest limit
    reg_v1 = Regulation(
        regulation_code="UN_R94",
        title="UNECE Standard UN_R94 04 Series",
        source_type="UNECE",
        amendment="04 Series",
        effective_date=datetime.date(2021, 1, 1),
        status="SUPERSEDED"
    )
    db_session.add(reg_v1)
    db_session.commit()

    doc_v1 = Document(
        regulation_id=reg_v1.id,
        document_name="UN_R94_04.pdf",
        document_type="PDF",
        hash="hash_v1",
        file_path="/storage/UNECE/UN_R94_04.pdf"
    )
    db_session.add(doc_v1)
    db_session.commit()

    chunk_v1 = Chunk(
        document_id=doc_v1.id,
        chunk_text="The Thorax Compression Criterion (ThCC) shall not exceed 42 mm;",
        chunk_index=1,
        page_number=12,
        section="5.2.1.4",
        chunk_type="clause"
    )
    db_session.add(chunk_v1)
    db_session.commit()

    # Version 2: 05 Series (effective 2026) with 50mm chest limit
    reg_v2 = Regulation(
        regulation_code="UN_R94",
        title="UNECE Standard UN_R94 05 Series",
        source_type="UNECE",
        amendment="05 Series",
        effective_date=datetime.date(2026, 1, 1),
        status="ACTIVE"
    )
    db_session.add(reg_v2)
    db_session.commit()

    doc_v2 = Document(
        regulation_id=reg_v2.id,
        document_name="UN_R94_05.pdf",
        document_type="PDF",
        hash="hash_v2",
        file_path="/storage/UNECE/UN_R94_05.pdf"
    )
    db_session.add(doc_v2)
    db_session.commit()

    chunk_v2 = Chunk(
        document_id=doc_v2.id,
        chunk_text="The Thorax Compression Criterion (ThCC) shall not exceed 50 mm;",
        chunk_index=1,
        page_number=12,
        section="5.2.1.4",
        chunk_type="clause"
    )
    db_session.add(chunk_v2)
    db_session.commit()

    # 2. Test Ingestion validation helper directly (Date 2025: resolves to v1 -> 42mm limit)
    date_2025 = datetime.date(2025, 6, 28)
    regs = db_session.query(Regulation).filter(Regulation.regulation_code == "UN_R94").all()
    
    # Resolve version
    applicable_regs = [r for r in regs if r.effective_date is not None and r.effective_date <= date_2025]
    applicable_regs.sort(key=lambda r: r.effective_date, reverse=True)
    resolved_reg = applicable_regs[0]
    assert resolved_reg.amendment == "04 Series"

    # Fetch chunk
    chunk = db_session.query(Chunk).join(Document).filter(
        Document.regulation_id == resolved_reg.id,
        Chunk.section == "5.2.1.4"
    ).first()
    assert chunk is not None
    assert "42 mm" in chunk.chunk_text

    # Verify criterion match
    assert verify_criterion_matches_clause("ThCC", chunk.chunk_text) is True
    assert verify_criterion_matches_clause("HIC36", chunk.chunk_text) is False

    # Extract limit & derive pass_fail (value 45mm vs 42mm limit -> FAIL)
    limit_details = extract_limit_details("ThCC", chunk.chunk_text)
    assert limit_details is not None
    limit, direction, unit = limit_details
    assert limit == 42.0
    assert (45.0 <= limit) is False  # FAIL

    # 3. Test Ingestion validation helper (Date 2026: resolves to v2 -> 50mm limit)
    date_2026 = datetime.date(2026, 6, 28)
    applicable_regs_2026 = [r for r in regs if r.effective_date is not None and r.effective_date <= date_2026]
    applicable_regs_2026.sort(key=lambda r: r.effective_date, reverse=True)
    resolved_reg_2026 = applicable_regs_2026[0]
    assert resolved_reg_2026.amendment == "05 Series"

    chunk_2026 = db_session.query(Chunk).join(Document).filter(
        Document.regulation_id == resolved_reg_2026.id,
        Chunk.section == "5.2.1.4"
    ).first()
    limit_details_2026 = extract_limit_details("ThCC", chunk_2026.chunk_text)
    assert limit_details_2026 is not None
    limit_2026, direction_2026, unit_2026 = limit_details_2026
    assert limit_2026 == 50.0
    assert (45.0 <= limit_2026) is True  # PASS


def test_query_time_margin_exact_transcripts(db_session):
    """Margin must be code-computed 42-38.4=3.6 mm for both exact transcript phrasings."""
    # Seed harness rows matching production ThCC measurements
    t1 = Test(
        test_id="TEST-2026-94-001",
        program="PLATFORM-X",
        date=datetime.date(2026, 6, 28),
        test_type="PHYSICAL_CRASH",
        impact_mode="FRONTAL_OFFSET",
        confidential_tier=True,
        owner_user_id="margin-user",
    )
    t2 = Test(
        test_id="TEST-PUB-FOCUS-1999",
        program="FOCUS",
        date=datetime.date(1999, 1, 1),
        test_type="PHYSICAL_CRASH",
        impact_mode="FRONTAL_OFFSET",
        confidential_tier=False,
    )
    db_session.add_all([t1, t2])
    db_session.commit()
    db_session.add_all(
        [
            TestResult(
                test_id="TEST-2026-94-001",
                injury_criterion="ThCC",
                value=38.4,
                pass_fail="PASS",
                linked_regulation_clause="UN_R94#5.2.1.4",
            ),
            TestResult(
                test_id="TEST-PUB-FOCUS-1999",
                injury_criterion="ThCC",
                value=34.2,
                pass_fail="PASS",
                linked_regulation_clause="UN_R94#5.2.1.4",
            ),
        ]
    )
    db_session.commit()

    expected = compute_headroom_mm(38.4, 42.0, "ceiling")
    assert expected == 3.6

    for transcript in (MARGIN_TRANSCRIPT_TURN_A, MARGIN_TRANSCRIPT_TURN_B):
        parsed = parse_measurement_pair(transcript)
        assert parsed is not None
        assert parsed.measured_mm == 38.4
        assert parsed.limit_mm == 42.0
        headroom = compute_headroom_mm(parsed.measured_mm, parsed.limit_mm, parsed.direction)
        assert headroom == 3.6

        out = try_margin_query_response(db_session, transcript, user_id="margin-user")
        assert out is not None
        assert out["metadata"]["response_route"] == "margin_compute"
        assert "**3.6 mm**" in out["answer"]
        assert "42.0 mm limit − 38.4 mm measured" in out["answer"]

    turn_b = try_margin_query_response(db_session, MARGIN_TRANSCRIPT_TURN_B, user_id="margin-user")
    assert turn_b is not None
    ans_b = turn_b["answer"]
    assert "harness test_results" in ans_b
    assert "TEST-2026-94-001" in ans_b
    assert "viscous criterion" not in ans_b.lower()
