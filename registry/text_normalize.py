"""Normalize regulation text for stable content hashing (FR-13)."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def content_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_extracted_text(text).encode("utf-8")).hexdigest()
