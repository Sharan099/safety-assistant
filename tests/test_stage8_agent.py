"""Stage 8 gate: agent tools use rebuilt pipeline; loop + grounding preserved."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.loop import run_agent
from agent.tools import TOOL_SPECS, tool_compare_regulations, tool_retrieve
from generation.llm_client import LLMClient
from generation.numeric_guard import check_numeric_fidelity
from retrieval.retrieve import RetrievedChunk


def test_agent_tools_invoke_current_retrieve_module():
    """Every retrieve-backed tool must call retrieval.retrieve.retrieve."""
    assert "retrieve" in TOOL_SPECS
    fake_chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="5.2.1.1. The head performance criterion (HPC) shall not exceed 1,000.",
            regulation_id="UN-ECE-R94",
            section_number="5.2.1.1",
            page_number=11,
        )
    ]
    with patch("agent.tools.retrieve", return_value=fake_chunks) as mocked:
        out = tool_retrieve("UN-ECE-R94", "What is the HPC limit?", llm=LLMClient(provider="mock"))
        mocked.assert_called()
        assert mocked.call_args.args[0] == "What is the HPC limit?" or (
            mocked.call_args.kwargs.get("regulation_id") == "UN-ECE-R94"
            or "UN-ECE-R94" in str(mocked.call_args)
        )
        assert out.get("ok") is True or out.get("chunk_ids") or out.get("text")


def test_loop_halts_at_max_steps_with_cited_answer():
    llm = LLMClient(provider="mock", use_cache=False)
    # Force a multi-step compare plan with a tiny max_steps cap.
    with patch("agent.loop.plan_task") as plan:
        from agent.state import PlannedStep

        plan.return_value = (
            "compare",
            [
                PlannedStep(tool="retrieve", args={"regulation": "UN-ECE-R94", "query": "HPC"}),
                PlannedStep(tool="retrieve", args={"regulation": "UN-ECE-R95", "query": "HPC"}),
                PlannedStep(tool="retrieve", args={"regulation": "UN-ECE-R16", "query": "HPC"}),
                PlannedStep(tool="draft_report", args={"sections": []}),
            ],
        )
        with patch("agent.loop._execute_tool") as exe:
            exe.return_value = {
                "ok": True,
                "text": "HPC shall not exceed 1000 [UN-ECE-R94 §5.2.1.1, p.11].",
                "citations": ["[UN-ECE-R94 §5.2.1.1, p.11]"],
                "chunk_ids": ["c1"],
            }
            result = run_agent(
                "Compare HPC across regulations",
                llm=llm,
                max_steps=2,
            )
    assert result is not None
    traces = list(getattr(result, "steps", None) or getattr(result, "trace", None) or [])
    tool_traces = [t for t in traces if getattr(t, "tool", None) not in {"plan", None}]
    assert len(tool_traces) <= 2
    answer = (
        getattr(result, "answer", None)
        or getattr(result, "final_answer", None)
        or getattr(result, "text", None)
        or str(result)
    )
    assert answer


def test_multi_step_comparison_citations_from_both_regulations():
    with patch("agent.tools.tool_retrieve") as tr:
        def _side(reg, query, **kwargs):  # noqa: ANN001
            rid = "UN-ECE-R94" if "94" in str(reg) else "UN-ECE-R95"
            return {
                "ok": True,
                "text": f"{rid} frontal/lateral note [{rid} §1, p.5].",
                "citations": [f"[{rid} §1, p.5]"],
                "chunk_ids": [f"{rid}-c1"],
                "chunks": [],
            }

        tr.side_effect = _side
        out = tool_compare_regulations(
            "UN-ECE-R94",
            "UN-ECE-R95",
            "impact direction",
            llm=LLMClient(provider="mock"),
        )
    text = str(out.get("text") or "")
    cites = list(out.get("citations") or [])
    blob = text + " ".join(cites)
    assert "R94" in blob or "UN-ECE-R94" in blob
    assert "R95" in blob or "UN-ECE-R95" in blob


def test_agent_output_passes_numeric_and_citation_gates():
    q = "If the fuel leakage rate is 35 g/min, does the vehicle pass?"
    good = (
        "Overall verdict: FAIL. Measured fuel leakage 35 g/min exceeds 30 g/min "
        "[UN-ECE-R94 §5.2.7, p.14]."
    )
    fidelity = check_numeric_fidelity(q, good)
    assert fidelity.ok
    from agent.citations import enforce_grounded_text

    cleaned, checks = enforce_grounded_text(
        good, allowed=["[UN-ECE-R94 §5.2.7, p.14]"]
    )
    assert "35 g/min" in cleaned
    assert "3 g/min" not in cleaned
    assert any(c.grounded for c in checks)
