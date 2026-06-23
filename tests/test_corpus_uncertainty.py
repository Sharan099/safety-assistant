"""Part F: disclosed corpus uncertainty flags."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.citations import detect_corpus_uncertainty_flags


def test_undisclosed_corpus_denial_flagged():
    flags = detect_corpus_uncertainty_flags(
        "Does UN R16 require a specific abrasion frequency?",
        "This requirement is not found in UN R16.",
    )
    assert any(f["type"] == "corpus_uncertainty_undisclosed" for f in flags)


def test_disclosed_uncertainty_not_flagged_as_undisclosed():
    flags = detect_corpus_uncertainty_flags(
        "Does UN R14 mention side-facing M3 loads?",
        "I am not confident this exists in UN R14 based on the retrieved passages.",
    )
    assert not any(f["type"] == "corpus_uncertainty_undisclosed" for f in flags)
