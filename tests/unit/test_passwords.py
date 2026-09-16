"""Password hashing (security/passwords.py): scrypt encoding, constant-time verification, and the
failure modes a leaked-hash or corrupted-row scenario would exercise."""

from __future__ import annotations

import pytest

from safety_assistant.security.passwords import hash_password, needs_rehash, verify_password


def test_hash_round_trips_and_is_not_the_plaintext() -> None:
    h = hash_password("correct horse battery staple")
    assert h.startswith("scrypt$")
    assert "correct horse battery staple" not in h
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong password", h)


def test_two_hashes_of_the_same_password_differ() -> None:
    # a random salt per hash defeats rainbow tables / cross-account hash comparison
    a, b = hash_password("same-password-1234"), hash_password("same-password-1234")
    assert a != b
    assert verify_password("same-password-1234", a)
    assert verify_password("same-password-1234", b)


@pytest.mark.parametrize(
    "bad",
    ["", "not-scrypt-at-all", "scrypt$x$8$1$salt$hash", "scrypt$16384$8$1$onlyfour$fields", "plaintext-password"],
)
def test_malformed_or_foreign_hash_fails_closed(bad: str) -> None:
    assert verify_password("anything", bad) is False


def test_empty_and_over_long_passwords_are_rejected() -> None:
    with pytest.raises(ValueError):
        hash_password("")
    with pytest.raises(ValueError):
        hash_password("x" * 257)
    assert verify_password("x" * 257, hash_password("short-enough")) is False


def test_needs_rehash_flags_weaker_or_foreign_parameters() -> None:
    current = hash_password("whatever-1234")
    assert needs_rehash(current) is False
    assert needs_rehash("scrypt$4096$8$1$c2FsdA$aGFzaA") is True  # weaker N
    assert needs_rehash("bcrypt$12$...") is True
    assert needs_rehash("garbage") is True
