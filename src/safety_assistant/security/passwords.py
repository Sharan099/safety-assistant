"""Password hashing (scrypt, stdlib — no dependency: `hashlib.scrypt` is OpenSSL's memory-hard KDF,
resistant to GPU/ASIC cracking if the database leaks). Never store or log a plaintext password.

Encoded format is self-describing so parameters can be strengthened later without invalidating
existing hashes: ``scrypt$N$r$p$<salt-b64>$<hash-b64>``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_N, _R, _P = 2**14, 8, 1  # ~16 MB, ~50-100 ms per hash on modern hardware (interactive login)
_SALT_BYTES = 16
_DKLEN = 32
_MAX_INPUT_BYTES = 256  # bound the KDF's cost; longer inputs than this add no real security


def hash_password(password: str) -> str:
    """Encoded hash for storage. Raises ValueError for an empty or unreasonably long password."""
    raw = password.encode("utf-8")
    if not raw or len(raw) > _MAX_INPUT_BYTES:
        raise ValueError("password length out of bounds")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(raw, salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN, maxmem=64 * 1024 * 1024)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check. False (never raises) for a malformed hash or an over-long password —
    a corrupt stored hash must fail closed, not error out and skip the check."""
    try:
        algo, n, r, p, salt_b64, digest_b64 = encoded.split("$")
        if algo != "scrypt":
            return False
        raw = password.encode("utf-8")
        if not raw or len(raw) > _MAX_INPUT_BYTES:
            return False
        salt, want = _unb64(salt_b64), _unb64(digest_b64)
        got = hashlib.scrypt(raw, salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(want), maxmem=64 * 1024 * 1024)
        return hmac.compare_digest(got, want)
    except (ValueError, TypeError):
        return False


def needs_rehash(encoded: str) -> bool:
    """True when a hash was made with weaker-than-current parameters (upgrade path after a
    parameter bump: re-hash silently on the next successful login)."""
    try:
        algo, n, r, p, _, _ = encoded.split("$")
        return algo != "scrypt" or (int(n), int(r), int(p)) != (_N, _R, _P)
    except ValueError:
        return True


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
