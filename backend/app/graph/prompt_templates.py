"""Mode-specific prompt supplements (templates referenced by config/modes.yaml)."""

from __future__ import annotations

PROMPT_TEMPLATES: dict[str, str] = {
    "clause_citation": (
        "Answer with exact clause citations [S#]. Quote numeric limits verbatim. "
        "Flag revision uncertainty when multiple revisions exist. "
        "Use [LEGAL] sources only for binding 'required/shall' claims."
    ),
    "design_review": (
        "Produce a design-review checklist table: Item | Status (Met/Affected/Needs revalidation) | "
        "Authority tier | Source [S#].\n"
        "Affected clauses: legal_binding [LEGAL] only. "
        "Affected tests: cite §6.4.x using test-configuration metadata. "
        "Kinematics risks: ONLY from ENG-REF or HISTORICAL — if absent, state sources not in corpus. "
        "Revalidation: apply 50mm tolerance exemption logic (R16 §7.7.1 / R14 §6.1.1.2) explicitly — "
        "compare stated displacement to 50mm before claiming exemption. "
        "Export-ready minutes format."
    ),
    "crash_compare": (
        "Compare measured values side-by-side. Highlight differing numerics. "
        "Cite test reports and regulation limits separately with authority tier badges."
    ),
    "rca_trace": (
        "STRUCTURED AUDITABLE CHAIN (trace view):\n"
        "1. Observation — state the failure fact; ASK for vehicle category if load/category ambiguous.\n"
        "2. Applicable legal threshold — [LEGAL] clause(s) only; which §6.4.x and limit exceeded/not-met.\n"
        "3. Probable causes — ONLY from HISTORICAL / ENG-REF / OEM chunks, each with [S#]. "
        "If none retrieved: 'No historical/engineering evidence retrieved — cannot infer causes.'\n"
        "4. Historical failures — HISTORICAL tier with program/test_id; note if different vehicle program.\n"
        "5. Affected clauses — legal_binding [LEGAL] citations.\n"
        "6. Recommended actions — advisory language only unless tied to legal non-compliance.\n"
        "Put [S#] at EACH step. Never invent causal mechanisms without retrieved support."
    ),
    "procedure_checklist": (
        "Output numbered test-prep steps. Link each step to its source clause [S#]."
    ),
    "post_test_compare": (
        "Compare measured results vs regulatory pass/fail thresholds in a table. "
        "Separate LEGAL limits from RATING thresholds."
    ),
    "browse_summary": (
        "Summarize relevant prior work across documents. Favor program-specific matches. "
        "Never present HISTORICAL or SYNTHETIC as regulatory requirements."
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
