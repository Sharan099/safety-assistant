"""Unit tests for agent citation gate + planner (no Qdrant required)."""

from __future__ import annotations

from agent.citations import check_claims, enforce_grounded_text, extract_citations
from agent.planner import detect_mode, plan_task
from agent.regs import normalize_regulation
from generation.llm_client import LLMClient


def test_normalize_regs():
    assert normalize_regulation("R94") == "UN-ECE-R94"
    assert normalize_regulation("fmvss 208") == "FMVSS-208"


def test_detect_modes():
    assert detect_mode("Compare R94 vs R95 chest deflection") == "compare"
    assert detect_mode("Gap analysis: test setup vs R95") == "report"
    assert detect_mode("What is HIC15 in R94?") == "qa"


def test_heuristic_plan_compare():
    mode, steps = plan_task("Compare R94 vs FMVSS 208 HIC limits", llm=LLMClient(provider="mock"))
    assert mode == "compare"
    assert steps[0].tool == "compare_regulations"


def test_citation_gate_strips_uncited():
    allowed = ["[UN-ECE-R94 §5.2.1, p.12]"]
    text = (
        "The HIC15 limit is 1000 [UN-ECE-R94 §5.2.1, p.12]. "
        "Aliens also require a limit of 42."
    )
    cleaned, checks = enforce_grounded_text(text, allowed=allowed)
    assert "Aliens" not in cleaned
    assert "1000" in cleaned
    assert extract_citations(cleaned)
    assert any(c.grounded for c in check_claims(cleaned, allowed=allowed))
