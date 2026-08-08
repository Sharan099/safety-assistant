"""Comparison-query generation instructions."""

from __future__ import annotations

from core.query_rewrite import QueryAnalysis


def comparison_user_instruction(analysis: QueryAnalysis) -> str:
    """User-message addendum for cross-regulation comparison questions."""
    regs = analysis.regulation_codes
    reg_list = ", ".join(regs) if regs else "each regulation present in Context"

    lines = [
        "COMPARISON MODE:",
        f"- Compare requirements across {reg_list}.",
        "- Use a markdown table when comparing two or more regulations.",
        "- Table columns: Feature | " + " | ".join(regs) if len(regs) >= 2 else "Feature | Regulation A | Regulation B",
        "- Put the direct comparison in the first row of substantive content (after an optional one-line lead).",
        "- Cite each cell's regulatory fact with [S#] and include regulation + clause (e.g. UN R94 §5.2.1.4).",
    ]
    if len(regs) >= 2:
        missing_note = (
            "- If Context lacks evidence for one regulation, state exactly which regulation or "
            "topic is missing — then complete the table for regulations that ARE covered."
        )
        lines.append(missing_note)
    return "\n".join(lines)
