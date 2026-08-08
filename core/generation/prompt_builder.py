"""Production prompt assembly for grounded answer generation."""

from __future__ import annotations

from core.generation.comparison_builder import comparison_user_instruction
from core.query_rewrite import QueryAnalysis, analyze_query
from security.prompts import (
    build_system_prompt,
    build_user_message,
    detect_answer_format,
)


def build_generation_messages(
    query: str,
    context: str,
    *,
    premise_notes: str = "",
    analysis: QueryAnalysis | None = None,
) -> list[dict[str, str]]:
    """Build system + user messages for the LLM router."""
    analysis = analysis or analyze_query(query)
    fmt = detect_answer_format(query, analysis)

    user_parts: list[str] = [
        build_user_message(context, query, answer_format=fmt),
    ]
    if premise_notes:
        user_parts.insert(1, premise_notes)
    if analysis.is_comparison:
        user_parts.insert(-1, comparison_user_instruction(analysis))

    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
