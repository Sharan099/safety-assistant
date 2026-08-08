"""Scoring logic per metric family."""

from eval.scoring.custom_checks import (
    CUSTOM_HARD_GATE_CATEGORIES,
    citation_grounding_check,
    cross_regulation_check,
    numeric_verbatim_check,
    run_custom_hard_gates,
)
from eval.scoring.ragas_scorer import (
    RAGAS_SCORE_CATEGORIES,
    score_case,
    score_cases,
    substring_checks,
)
from eval.scoring.security_scorer import (
    SECURITY_CATEGORIES,
    answer_has_not_found_pattern,
    expects_not_found,
    run_security_scorer,
    score_security_suite,
)

__all__ = [
    "CUSTOM_HARD_GATE_CATEGORIES",
    "RAGAS_SCORE_CATEGORIES",
    "SECURITY_CATEGORIES",
    "answer_has_not_found_pattern",
    "citation_grounding_check",
    "cross_regulation_check",
    "expects_not_found",
    "numeric_verbatim_check",
    "run_custom_hard_gates",
    "run_security_scorer",
    "score_case",
    "score_cases",
    "score_security_suite",
    "substring_checks",
]
