"""Copilot API — streaming (real-time workflow visibility) and history."""

import json
import re
from typing import Any

from fastapi.testclient import TestClient

from apps.api.main import app
from tests.conftest import requires_db

client = TestClient(app)


def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_match = re.search(r"^event: (.+)$", block, re.MULTILINE)
        data_match = re.search(r"^data: (.+)$", block, re.MULTILINE)
        if event_match and data_match:
            events.append((event_match.group(1), json.loads(data_match.group(1))))
    return events


def _create_scn001_investigation_and_run_agent() -> str:
    create = client.post(
        "/api/v1/investigations",
        json={
            "run_a_id": "SCN-001-RUN-A",
            "run_b_id": "SCN-001-RUN-B",
            "question": "Why did chest deflection increase?",
            "primary_metric": "chest_deflection",
        },
    )
    investigation_id: str = create.json()["id"]
    run = client.post(f"/api/v1/investigations/{investigation_id}/run-agent")
    assert run.status_code == 200, run.text
    return investigation_id


@requires_db
def test_copilot_message_streams_steps_then_final() -> None:
    investigation_id = _create_scn001_investigation_and_run_agent()

    resp = client.post(
        f"/api/v1/investigations/{investigation_id}/copilot/messages",
        json={"message": "Why is the restraint configuration currently a leading contributor?"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    event_types = [e[0] for e in events]
    assert event_types == ["step", "step", "step", "step", "final"]

    final_data = events[-1][1]
    assert final_data["intent"] == "EXPLAIN_EVIDENCE"
    assert final_data["message"]
    assert isinstance(final_data["evidence_refs"], list)
    assert isinstance(final_data["tool_activity"], list)


@requires_db
def test_copilot_message_error_for_unknown_investigation() -> None:
    import uuid

    resp = client.post(
        f"/api/v1/investigations/{uuid.uuid4()}/copilot/messages",
        json={"message": "hello"},
    )
    assert resp.status_code == 200  # SSE stream itself succeeds; error is an event
    events = _parse_sse(resp.text)
    assert events[0][0] == "error"
    assert "not found" in events[0][1]["detail"]


@requires_db
def test_copilot_message_history_persisted_and_listable() -> None:
    investigation_id = _create_scn001_investigation_and_run_agent()

    client.post(
        f"/api/v1/investigations/{investigation_id}/copilot/messages",
        json={"message": "Compare the crash pulse"},
    )

    history = client.get(f"/api/v1/investigations/{investigation_id}/copilot/messages")
    assert history.status_code == 200
    messages = history.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Compare the crash pulse"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_activity"]
    assert messages[1]["tool_activity"][0]["tool_name"] == "compare_global_response"


@requires_db
def test_copilot_history_empty_for_investigation_with_no_conversation() -> None:
    create = client.post(
        "/api/v1/investigations",
        json={"run_a_id": "SCN-002-RUN-A", "run_b_id": "SCN-002-RUN-B", "question": "x"},
    )
    investigation_id = create.json()["id"]
    history = client.get(f"/api/v1/investigations/{investigation_id}/copilot/messages")
    assert history.status_code == 200
    assert history.json() == []
