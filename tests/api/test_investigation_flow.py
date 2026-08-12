"""End-to-end V1 vertical slice over HTTP — APP_FLOW.md §22.

Run A/B -> quality -> comparability -> configuration -> chest-deflection
signal -> divergence, against the real SCN-001 synthetic runs already loaded
by `scripts/generate_synthetic_dataset.py`.
"""

from fastapi.testclient import TestClient

from apps.api.main import app
from tests.domain.conftest import requires_db

client = TestClient(app)


@requires_db
def test_health() -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@requires_db
def test_list_and_get_run() -> None:
    resp = client.get("/api/v1/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert any(r["run_id"] == "SCN-001-RUN-A" for r in runs)

    resp = client.get("/api/v1/runs/SCN-001-RUN-A")
    assert resp.status_code == 200
    assert resp.json()["quality_status"] == "PASS"
    assert resp.json()["model_version"] == "v12.3"

    resp = client.get("/api/v1/runs/does-not-exist")
    assert resp.status_code == 404


@requires_db
def test_scn001_full_vertical_slice() -> None:
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
    assert create.json()["state"] == "RUNS_SELECTED"

    quality = client.post(f"/api/v1/investigations/{investigation_id}/quality")
    assert quality.status_code == 200
    assert quality.json()["run_a"]["overall_status"] == "PASS"
    assert quality.json()["run_b"]["overall_status"] == "PASS"

    global_response = client.post(f"/api/v1/investigations/{investigation_id}/global-response")
    assert global_response.status_code == 200
    assert global_response.json()["material_difference_detected"] is False

    config_diff = client.post(f"/api/v1/investigations/{investigation_id}/configuration-diff")
    assert config_diff.status_code == 200
    changed = {d["path"] for d in config_diff.json() if d["change_status"] == "CHANGED"}
    assert "restraint.force_limiter.level_n" in changed

    comparability = client.post(f"/api/v1/investigations/{investigation_id}/comparability")
    assert comparability.status_code == 200
    dims = {d["dimension"]: d["status"] for d in comparability.json()["dimensions"]}
    assert dims["global_pulse"] == "COMPARABLE"
    assert dims["causal_isolation"] == "COMPARABLE"

    signal = client.post(f"/api/v1/investigations/{investigation_id}/signals/chest_deflection/analyze")
    assert signal.status_code == 200
    assert signal.json()["divergence"] is not None

    final = client.get(f"/api/v1/investigations/{investigation_id}")
    assert final.status_code == 200
    assert final.json()["state"] == "SIGNAL_ANALYSIS"


@requires_db
def test_scn010_quality_failure_surfaces_over_http() -> None:
    create = client.post(
        "/api/v1/investigations",
        json={"run_a_id": "SCN-010-RUN-A", "run_b_id": "SCN-010-RUN-B", "question": "Did Run B complete normally?"},
    )
    investigation_id = create.json()["id"]

    quality = client.post(f"/api/v1/investigations/{investigation_id}/quality")
    assert quality.json()["run_b"]["overall_status"] == "FAIL"

    comparability = client.post(f"/api/v1/investigations/{investigation_id}/comparability")
    assert comparability.json()["overall_status"] == "NOT_COMPARABLE"
