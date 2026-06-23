"""Mode-specific prompt supplements (templates referenced by config/modes.yaml)."""

from __future__ import annotations

PROMPT_TEMPLATES: dict[str, str] = {
    "clause_citation": (
        "Answer with exact clause citations [S#]. Quote numeric limits verbatim. "
        "Flag revision uncertainty when multiple revisions exist."
    ),
    "design_review": (
        "Present requirement-by-requirement status: Met / Not met / Needs review. "
        "Use a table when multiple requirements are in context."
    ),
    "crash_compare": (
        "Compare measured values side-by-side. Highlight differing numerics. "
        "Cite test reports and regulation limits separately."
    ),
    "rca_trace": (
        "Structure as auditable chain: Observation → Evidence → Reasoning → Conclusion. "
        "Put [S#] at EACH step. Do not skip causal links."
    ),
    "procedure_checklist": (
        "Output numbered test-prep steps. Link each step to its source clause [S#]."
    ),
    "post_test_compare": (
        "Compare measured results vs regulatory pass/fail thresholds in a table."
    ),
    "browse_summary": (
        "Summarize relevant prior work across documents. Favor program-specific matches."
    ),
    "executive_summary": (
        "Executive summary for management: NO clause numbers in the main answer. "
        "Use plain language, risks, and milestones. Put clause detail only if user drills in."
    ),
}


def template_for(mode_name: str | None) -> str:
    from backend.app.core.modes import get_mode

    mode = get_mode(mode_name)
    return PROMPT_TEMPLATES.get(
        mode.prompt_template_name,
        PROMPT_TEMPLATES["clause_citation"],
    )
