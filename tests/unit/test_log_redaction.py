"""Operational logs never carry credentials (connection-string passwords, bearer tokens, key=value secrets)."""

from __future__ import annotations

from safety_assistant.observability.logging import redact


def test_redacts_connection_strings_tokens_and_key_values() -> None:
    assert redact("postgresql+psycopg://sa:Sup3r-secret@db:5432/x") == "postgresql+psycopg://sa:***@db:5432/x"
    assert redact("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def") == "Authorization: Bearer ***"
    assert redact('{"api_key": "sk-abcdefghijkl"}') == '{"api_key": "***"}'
    assert redact("plain message with no secrets, status=200") == "plain message with no secrets, status=200"
