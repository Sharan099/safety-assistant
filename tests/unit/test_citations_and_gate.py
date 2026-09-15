import datetime
import uuid

from safety_assistant.generation import GroundedDraft, rewrite_query, validate_draft
from safety_assistant.generation.citations import canonical_number, claimed_numbers
from safety_assistant.generation.schemas import Claim
from safety_assistant.retrieval.context import Evidence, LegRanks


def _ev(eid: str, content: str) -> Evidence:
    return Evidence(
        evidence_id=eid,
        chunk_id=uuid.uuid4(),
        regulation_key="UN-R94",
        regulation_title="t",
        kind="REGULATION",
        jurisdiction="UNECE",
        authority_level="AUTHORITATIVE",
        version_id=uuid.uuid4(),
        version_label="Rev.4",
        version_status="ACTIVE",
        valid_from=datetime.date(2021, 6, 9),
        valid_to=None,
        published_at=None,
        section_id=uuid.uuid4(),
        section_path="5.2.1.8",
        section_number="5.2.1.8.",
        section_title=None,
        annex=None,
        normative=True,
        chunk_type="TEXT",
        page_start=13,
        page_end=13,
        citation_label="UN R94 Rev.4 §5.2.1.8 (p. 13)",
        content=content,
        ranks=LegRanks(fused_score=0.1),
        source_sha256="x" * 64,
        source_uri=None,
        storage_uri="file://x",
        token_count=10,
    )


def test_canonical_numbers() -> None:
    assert canonical_number("1,3") == "1.3" and canonical_number("1,000") == "1000"
    assert canonical_number("3,500") == "3500" and canonical_number("1.25") == "1.25"
    assert claimed_numbers("exceed 1,3; HPC 1,000; item 1") == {"1.3", "1000"}


def test_validation_accepts_supported_and_rejects_invented_numbers_and_ids() -> None:
    ev = [_ev("E1", "The tibia index (TI) shall not exceed 1,3 at either location.")]
    draft = GroundedDraft(
        answer="x",
        claims=[
            Claim(text="The tibia index shall not exceed 1.3.", evidence_ids=["E1"]),
            Claim(text="The tibia index shall not exceed 1.5.", evidence_ids=["E1"]),
            Claim(text="Stated in clause 9.", evidence_ids=["E9"]),
            Claim(text="This is my reading with number 77.", evidence_ids=["E1"], kind="INTERPRETATION"),
        ],
    )
    kept, report = validate_draft(draft, ev)
    assert [c.text for c in kept] == ["The tibia index shall not exceed 1.3.", "This is my reading with number 77."]
    statuses = [c.status for c in report.claims]
    assert statuses == ["SUPPORTED", "NUMERIC_MISMATCH", "UNSUPPORTED_EVIDENCE_ID", "SUPPORTED"]
    assert report.unknown_evidence_ids == ["E9"] and report.dropped_claims == 2 and report.ok is False


def test_validation_accepts_numbers_from_evidence_attributes_but_not_invented_values() -> None:
    ev = _ev("E1", "The strap shall be kept for three hours in a heating cabinet at 60 + 5 °C.")
    e = ev.model_copy(update={"section_path": "7.4.1.4.1", "version_label": "Rev.7 (06 series)"})
    draft = GroundedDraft(
        answer="x",
        claims=[
            Claim(text="Per §7.4.1.4.1 of Rev.7, the strap is kept at 60 + 5 °C.", evidence_ids=["E1"]),
            Claim(text="Per §7.4.1.4.1 of Rev.7, the strap is kept at 80 °C.", evidence_ids=["E1"]),
        ],
    )
    kept, report = validate_draft(draft, [e])
    assert [c.status for c in report.claims] == ["SUPPORTED", "NUMERIC_MISMATCH"]
    assert len(kept) == 1


def test_rewrite_expands_known_acronyms() -> None:
    assert rewrite_query("HPC limit in R94") == "HPC (head performance criterion) limit in R94"
    assert rewrite_query("frontal collision") is None


def test_calculation_claims_may_derive_numbers_but_must_start_from_evidence() -> None:
    ev = [_ev("E1", "Vehicle speed at the moment of impact shall be 56 -0/+1 km/h.")]
    derived = Claim(text="56 km/h is 15.6 m/s (56 / 3.6).", evidence_ids=["E1"], kind="CALCULATION")
    invented = Claim(text="At 64 km/h the energy is 17.8 m/s squared.", evidence_ids=["E1"], kind="CALCULATION")
    as_requirement = Claim(text="56 km/h is 15.6 m/s.", evidence_ids=["E1"], kind="REQUIREMENT")
    kept, report = validate_draft(GroundedDraft(answer="", claims=[derived, invented, as_requirement]), ev)
    assert [c.kind for c in kept] == [
        "CALCULATION"
    ]  # derived result allowed, invented inputs and REQUIREMENT conversions are not
    assert [r.status for r in report.claims] == ["SUPPORTED", "NUMERIC_MISMATCH", "NUMERIC_MISMATCH"]


def test_small_talk_gets_a_capabilities_reply_without_retrieval() -> None:
    from safety_assistant.generation.grounding import CAPABILITIES, small_talk

    assert small_talk("Hi!") == CAPABILITIES
    assert small_talk("what can you do?") == CAPABILITIES
    assert small_talk("What is the ThCC limit?") is None


def test_padded_citations_are_pruned_to_the_supporting_evidence() -> None:
    ev = [
        _ev("E1", "5.2.1.4. The Thorax Compression Criterion (ThCC) shall not exceed 42 mm;"),
        _ev("E2", "5.2.1.2. Rib Deflection Criterion (RDC) less than or equal to 42 mm;"),
        _ev("E3", "2.1. Protective system means the interior fittings and devices intended to restrain the occupants."),
    ]
    numeric = Claim(text="The ThCC limit is 42 mm.", evidence_ids=["E1", "E2", "E3"])
    kept, _ = validate_draft(GroundedDraft(answer="", claims=[numeric]), ev)
    assert set(kept[0].evidence_ids) == {"E1", "E2"}  # both state 42 mm; the unrelated definition is dropped
    prose = Claim(text="A protective system restrains the occupants.", evidence_ids=["E1", "E3"])
    kept, _ = validate_draft(GroundedDraft(answer="", claims=[prose]), ev)
    assert kept[0].evidence_ids == ["E3"]


def test_numbers_stated_in_the_question_are_known_inputs() -> None:
    ev = [_ev("E1", "5.2.1.4. The Thorax Compression Criterion (ThCC) shall not exceed 42 mm;")]
    claim = Claim(text="Your 44 mm exceeds the 42 mm ThCC limit.", evidence_ids=["E1"])
    kept, _ = validate_draft(GroundedDraft(answer="", claims=[claim]), ev)
    assert kept == []  # 44 mm comes from nowhere
    kept, _ = validate_draft(
        GroundedDraft(answer="", claims=[claim]), ev, question="Our result is 44 mm; does it pass?"
    )
    assert len(kept) == 1
    invented = Claim(text="Your 44 mm exceeds the 45 mm ThCC limit.", evidence_ids=["E1"])
    kept, _ = validate_draft(
        GroundedDraft(answer="", claims=[invented]), ev, question="Our result is 44 mm; does it pass?"
    )
    assert kept == []  # the limit itself must still come from the evidence
