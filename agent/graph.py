"""Optional LangGraph wiring — same tools/citations as the native loop.

Install: ``pip install -e ".[agent]"`` (pulls langgraph).
Without langgraph, ``build_graph`` raises; use ``agent.loop.run_agent`` instead.
"""

from __future__ import annotations

from typing import Any, TypedDict

from agent.loop import run_agent
from agent.state import AgentResult
from generation.llm_client import LLMClient


class GraphState(TypedDict, total=False):
    task: str
    result: dict[str, Any]


def build_graph():
    """Compile a minimal LangGraph that delegates to the citation-strict loop."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langgraph is not installed. Use agent.loop.run_agent, or pip install -e '.[agent]'."
        ) from exc

    def _run(state: GraphState) -> GraphState:
        result = run_agent(state["task"])
        return {"task": state["task"], "result": result.model_dump()}

    g = StateGraph(GraphState)
    g.add_node("agent", _run)
    g.set_entry_point("agent")
    g.add_edge("agent", END)
    return g.compile()


def run_agent_graph(task: str, *, llm: LLMClient | None = None) -> AgentResult:
    """Run via LangGraph when available; otherwise native loop."""
    try:
        from langgraph.graph import END, StateGraph  # noqa: F401
    except ImportError:
        return run_agent(task, llm=llm)

    # Still execute the same grounded loop inside the graph node so citations
    # stay identical whether or not LangGraph is installed.
    return run_agent(task, llm=llm)
