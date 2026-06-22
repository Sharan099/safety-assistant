"""Output sanitizer and injury-value guard tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.guardrails.output_sanitizer import (  # noqa: E402
    contains_injury_increase_advice,
    sanitize_model_output,
)


def test_strips_template_leak():
    raw = "The limit is 34 mm <|start_header_id|>assistant"
    clean, warnings = sanitize_model_output(raw)
    assert "<|redacted" not in clean
    assert "template_leak_stripped" in warnings


def test_blocks_injury_increase():
    raw = "Recommend increasing chest deflection to ≥ 50 mm for better performance."
    clean, warnings = sanitize_model_output(raw)
    assert "injury_increase_blocked" in warnings
    assert "cannot recommend increasing" in clean.lower()


def test_normal_answer_unchanged():
    raw = "Chest deflection limit is 34 mm [S1]."
    clean, warnings = sanitize_model_output(raw)
    assert clean == raw
    assert not warnings


def test_contains_injury_increase_detector():
    assert contains_injury_increase_advice("improve chest deflection to 50 mm")
    assert not contains_injury_increase_advice("reduce chest deflection to 30 mm")
