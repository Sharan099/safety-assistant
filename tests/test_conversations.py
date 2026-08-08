"""Conversation store + follow-up condensation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from api.conversations import (
    Turn,
    append_turn,
    clear_conversation,
    get_turns,
    init_db,
    new_conversation_id,
)
from retrieval.rewrite import condense_followup, rewrite_query


@pytest.fixture()
def conv_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "conversations.sqlite3"
    monkeypatch.setenv("CONVERSATIONS_DB", str(db))
    monkeypatch.setenv("CONVERSATION_MAX_TURNS", "6")
    # Reset module-level path cache.
    import api.conversations as conv

    conv._db_path = None
    init_db()
    yield db
    conv._db_path = None


def test_conversation_trims_to_max_turns(conv_db: Path):
    cid = new_conversation_id()
    for i in range(8):
        append_turn(cid, f"q{i}", f"a{i}")
    turns = get_turns(cid)
    assert len(turns) == 6
    assert turns[0].question == "q2"
    assert turns[-1].question == "q7"


def test_condense_skips_without_history():
    q = "What is the HPC limit for frontal impact?"
    condensed, applied = condense_followup(q, [], use_llm=False)
    assert condensed == q
    assert applied is False


def test_condense_resolves_neck_followup():
    history = [
        Turn(
            question="What is the HPC limit for frontal impact under UN-ECE-R94?",
            answer="HPC shall not exceed 1000.",
        )
    ]
    condensed, applied = condense_followup(
        "What about the neck injury criterion?",
        history,
        use_llm=False,
    )
    assert applied is True
    low = condensed.lower()
    assert "neck injury" in low
    assert any(tok in condensed for tok in ("UN-ECE-R94", "frontal", "HPC"))


def test_rewrite_logs_original_and_condensed():
    history = [
        Turn(
            question="What is the ThCC limit in UN R94 frontal impact?",
            answer="42 mm",
        )
    ]
    result = rewrite_query("What about the VC limit?", use_llm=False, history=history)
    assert result.original == "What about the VC limit?"
    assert result.condensation_applied is True
    assert "VC" in result.condensed
    assert any(tok in result.condensed for tok in ("UN-ECE-R94", "R94", "frontal", "ThCC"))


def test_clear_conversation(conv_db: Path):
    cid = new_conversation_id()
    append_turn(cid, "q", "a")
    clear_conversation(cid)
    assert get_turns(cid) == []
