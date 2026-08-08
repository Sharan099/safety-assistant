"""Generation-stage building blocks — prompts, context, comparison, answer formatting."""

from core.generation.answer_formatter import format_answer_for_display, strip_scoring_artifacts
from core.generation.comparison_builder import comparison_user_instruction
from core.generation.context_builder import prepare_llm_context
from core.generation.prompt_builder import build_generation_messages

__all__ = [
    "build_generation_messages",
    "comparison_user_instruction",
    "format_answer_for_display",
    "prepare_llm_context",
    "strip_scoring_artifacts",
]
