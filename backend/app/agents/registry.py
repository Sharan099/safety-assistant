"""Pre-configured specialist agents for the crash-development crew."""

from __future__ import annotations

from backend.app.agents.base import load_prompt, make_agent
from backend.app.agents.schemas import CrewState
from backend.app.core.authority_tier import (
    ENGINEERING_REF,
    HISTORICAL_DATA,
    LEGAL_BINDING,
    SYNTHETIC,
)


def _crash_block(state: CrewState) -> str:
    parts = [state.get("crash_input", "")]
    if state.get("vehicle"):
        parts.append(f"Vehicle: {state['vehicle']}")
    if state.get("crash_summary"):
        parts.append(f"Summary: {state['crash_summary']}")
    return "\n".join(p for p in parts if p)


def build_simulation_agent():
    return make_agent(
        "simulation",
        mode="post_test_analysis",
        allowed_tiers=(HISTORICAL_DATA, SYNTHETIC),
        tier=1,
        system_prompt=load_prompt("simulation"),
        build_query=lambda s: f"crash test metrics post-test analysis {_crash_block(s)}",
        build_user=lambda s, q, ctx: (
            f"Parse this crash result table into structured metrics.\n\n"
            f"{_crash_block(s)}\n\nRetrieved context:\n{ctx}"
        ),
    )


def build_regulation_agent():
    return make_agent(
        "regulation",
        mode="regulation_lookup",
        allowed_tiers=(LEGAL_BINDING,),
        tier=2,
        system_prompt=load_prompt("regulation"),
        build_query=lambda s: (
            f"regulatory limits injury criteria chest deflection HIC femur "
            f"{_crash_block(s)}"
        ),
        build_user=lambda s, q, ctx: (
            f"Map failing metrics to legal_binding regulatory limits.\n\n"
            f"Crash data:\n{_crash_block(s)}\n\n"
            f"Prior simulation output:\n{s.get('agent_outputs', {}).get('simulation', {})}\n\n"
            f"Sources (legal_binding only):\n{ctx}"
        ),
    )


def build_root_cause_agent():
    return make_agent(
        "root_cause",
        mode="root_cause_analysis",
        allowed_tiers=(LEGAL_BINDING, ENGINEERING_REF, HISTORICAL_DATA, SYNTHETIC),
        tier=3,
        system_prompt=load_prompt("root_cause"),
        build_query=lambda s: (
            f"root cause analysis crash structural restraint {_crash_block(s)}"
        ),
        build_user=lambda s, q, ctx: (
            f"Hypothesize root causes for failing metrics.\n\n"
            f"Crash data:\n{_crash_block(s)}\n\n"
            f"Simulation:\n{s.get('agent_outputs', {}).get('simulation', {})}\n\n"
            f"Regulation:\n{s.get('agent_outputs', {}).get('regulation', {})}\n\n"
            f"Sources:\n{ctx}"
        ),
    )


def build_knowledge_agent():
    return make_agent(
        "knowledge",
        mode="knowledge_reuse",
        allowed_tiers=(HISTORICAL_DATA, SYNTHETIC),
        tier=2,
        system_prompt=load_prompt("knowledge"),
        build_query=lambda s: (
            f"similar crash test cases historical knowledge reuse {_crash_block(s)}"
        ),
        build_user=lambda s, q, ctx: (
            f"Find similar historical/synthetic cases.\n\n"
            f"Crash data:\n{_crash_block(s)}\n\n"
            f"Root causes:\n{s.get('agent_outputs', {}).get('root_cause', {})}\n\n"
            f"Sources:\n{ctx}"
        ),
    )


def build_countermeasure_agent():
    return make_agent(
        "countermeasure",
        mode="design_review",
        allowed_tiers=(ENGINEERING_REF, SYNTHETIC),
        tier=3,
        system_prompt=load_prompt("countermeasure"),
        build_query=lambda s: (
            f"design countermeasure crash restraint {_crash_block(s)}"
        ),
        build_user=lambda s, q, ctx: (
            f"Propose ranked countermeasures.\n\n"
            f"Crash data:\n{_crash_block(s)}\n\n"
            f"Root causes:\n{s.get('agent_outputs', {}).get('root_cause', {})}\n\n"
            f"Similar cases:\n{s.get('agent_outputs', {}).get('knowledge', {})}\n\n"
            f"Sources:\n{ctx}"
        ),
    )


def build_program_manager_agent():
    return make_agent(
        "program_manager",
        mode="management_view",
        allowed_tiers=None,
        tier=2,
        system_prompt=load_prompt("program_manager"),
        build_query=lambda s: f"executive crash program summary {_crash_block(s)}",
        build_user=lambda s, q, ctx: (
            f"Synthesize all agent outputs into a management report.\n\n"
            f"Crash data:\n{_crash_block(s)}\n\n"
            f"All agent outputs:\n{s.get('agent_outputs', {})}\n"
        ),
        skip_retrieval=True,
    )
