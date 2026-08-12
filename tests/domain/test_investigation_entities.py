"""Round-trip the evidence chain: Investigation -> Evidence -> Hypothesis ->
HypothesisEvidenceLink -> EngineerReview.

Guards BACKEND_SCHEMA.md §50 invariant 6: supporting and contradicting
evidence are separate relationships, never one boolean.
"""

from sqlalchemy.orm import Session

from packages.domain.core import Organization, Project, User
from packages.domain.investigation import (
    EngineerReview,
    Evidence,
    Hypothesis,
    HypothesisEvidenceLink,
    Investigation,
)
from tests.domain.conftest import requires_db


@requires_db
def test_evidence_hypothesis_review_chain(session: Session) -> None:
    org = Organization(name="ACME Automotive")
    session.add(org)
    session.flush()

    user = User(organization_id=org.id, email="reviewer@example.com", display_name="R. Reviewer", role="REVIEWER")
    project = Project(organization_id=org.id, name="Project X")
    session.add_all([user, project])
    session.flush()

    investigation = Investigation(
        project_id=project.id,
        created_by=user.id,
        title="Chest deflection increase RUN-0040 vs RUN-0041",
        question="Why did chest deflection increase between Run A and Run B?",
        state="HYPOTHESIS_ANALYSIS",
    )
    session.add(investigation)
    session.flush()

    supporting = Evidence(
        investigation_id=investigation.id,
        evidence_type="CALCULATED",
        source_type="signal_analysis",
        content="Belt force diverged at 41.7 ms",
        calculation_version="first_divergence_detector v0.1.0",
    )
    contradicting = Evidence(
        investigation_id=investigation.id,
        evidence_type="OBSERVED",
        source_type="configuration_diff",
        content="Vehicle model revision also changed between runs",
    )
    session.add_all([supporting, contradicting])
    session.flush()

    hypothesis = Hypothesis(
        investigation_id=investigation.id,
        title="Changed belt behaviour is a leading contributor",
        status="PARTIALLY_SUPPORTED",
        created_by="agent",
    )
    session.add(hypothesis)
    session.flush()

    links = [
        HypothesisEvidenceLink(hypothesis_id=hypothesis.id, evidence_id=supporting.id, relationship="SUPPORTS"),
        HypothesisEvidenceLink(hypothesis_id=hypothesis.id, evidence_id=contradicting.id, relationship="CONTRADICTS"),
    ]
    session.add_all(links)
    session.flush()

    review = EngineerReview(
        investigation_id=investigation.id,
        reviewer_id=user.id,
        decision="MODIFY",
        comment="Request controlled belt isolation before accepting.",
    )
    session.add(review)
    session.flush()

    session.expire_all()

    reloaded_links = (
        session.query(HypothesisEvidenceLink)
        .filter_by(hypothesis_id=hypothesis.id)
        .order_by(HypothesisEvidenceLink.relationship)
    ).all()
    relationships = {link.relationship for link in reloaded_links}
    assert relationships == {"SUPPORTS", "CONTRADICTS"}

    reloaded_review = session.get(EngineerReview, review.id)
    assert reloaded_review is not None
    assert reloaded_review.decision == "MODIFY"
