"""End-to-end: create investigation -> run agent -> engineer review, over HTTP."""

from fastapi.testclient import TestClient

from apps.api.main import app
from tests.legacy.conftest import requires_db

client = TestClient(app)


@requires_db
def test_scn001_agent_run_and_review_over_http() -> None:
    create = client.post(
        "/api/v1/investigations",
        json={
            "run_a_id": "SCN-001-RUN-A",
            "run_b_id": "SCN-001-RUN-B",
            "question": "Why did chest deflection increase?",
            "primary_metric": "chest_deflection",
        },
    )
    assert create.status_code == 201, create.text
    investigation_id = create.json()["id"]
    assert create.json()["primary_metric"] == "chest_deflection"

    run = client.post(f"/api/v1/investigations/{investigation_id}/run-agent")
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["state"] == "ENGINEER_REVIEW"
    assert body["blocked_reason"] is None
    assert len(body["hypotheses"]) == 1
    assert body["evidence_count"] > 0

    evidence = client.get(f"/api/v1/investigations/{investigation_id}/evidence")
    assert evidence.status_code == 200
    assert len(evidence.json()) == body["evidence_count"]

    hypotheses = client.get(f"/api/v1/investigations/{investigation_id}/hypotheses")
    assert hypotheses.status_code == 200
    assert len(hypotheses.json()) == 1

    review = client.post(
        f"/api/v1/investigations/{investigation_id}/review",
        json={"decision": "MODIFY", "comment": "Request controlled belt isolation before accepting."},
    )
    assert review.status_code == 201
    assert review.json()["investigation_state"] == "FOLLOW_UP"

    final = client.get(f"/api/v1/investigations/{investigation_id}")
    assert final.json()["state"] == "FOLLOW_UP"


@requires_db
def test_scn010_agent_run_blocks_over_http() -> None:
    create = client.post(
        "/api/v1/investigations",
        json={"run_a_id": "SCN-010-RUN-A", "run_b_id": "SCN-010-RUN-B", "question": "Did Run B complete normally?"},
    )
    investigation_id = create.json()["id"]

    run = client.post(f"/api/v1/investigations/{investigation_id}/run-agent")
    assert run.status_code == 200
    body = run.json()
    assert body["state"] == "BLOCKED"
    assert body["blocked_reason"] == "quality_gate_failed"


@requires_db
def test_unknown_primary_metric_rejected() -> None:
    resp = client.post(
        "/api/v1/investigations",
        json={
            "run_a_id": "SCN-001-RUN-A",
            "run_b_id": "SCN-001-RUN-B",
            "question": "x",
            "primary_metric": "not_a_signal",
        },
    )
    assert resp.status_code == 422


@requires_db
def test_review_rejects_invalid_decision() -> None:
    create = client.post(
        "/api/v1/investigations",
        json={"run_a_id": "SCN-001-RUN-A", "run_b_id": "SCN-001-RUN-B", "question": "x"},
    )
    investigation_id = create.json()["id"]
    resp = client.post(f"/api/v1/investigations/{investigation_id}/review", json={"decision": "NOT_A_REAL_DECISION"})
    assert resp.status_code == 422
