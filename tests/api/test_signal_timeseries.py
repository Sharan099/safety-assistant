from fastapi.testclient import TestClient

from apps.api.main import app
from tests.conftest import requires_db

client = TestClient(app)


@requires_db
def test_signal_timeseries_returns_raw_arrays() -> None:
    create = client.post(
        "/api/v1/investigations",
        json={
            "run_a_id": "SCN-001-RUN-A",
            "run_b_id": "SCN-001-RUN-B",
            "question": "x",
            "primary_metric": "chest_deflection",
        },
    )
    investigation_id = create.json()["id"]

    resp = client.get(f"/api/v1/investigations/{investigation_id}/signals/chest_deflection/timeseries")
    assert resp.status_code == 200
    body = resp.json()
    assert body["signal"] == "chest_deflection"
    assert len(body["time_s"]) == len(body["run_a"]) == len(body["run_b"])
    assert len(body["time_s"]) > 0
    assert body["unit"] == "mm"
