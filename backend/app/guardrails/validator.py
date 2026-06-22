"""
Guardrails: prompt injection, jailbreak, PII leakage, unsafe content.
Uses pattern checks + optional guardrails-ai hub validators.

Three input outcomes (Phase 1):
  - injection_blocked — genuine prompt-injection / jailbreak only
  - answerable        — normal regulatory questions (including comparisons)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

InputState = Literal["injection_blocked", "answerable"]

# Genuine injection — instruction override directed at the model, not domain language.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|the\s+)?(system|above)\s+instructions",
    r"you\s+are\s+now\s+(dan|evil|unrestricted|jailbroken)",
    r"pretend\s+you\s+are\s+not\s+(an?\s+)?(ai|assistant|chatbot)",
    r"\bjailbreak\b",
    r"\bdeveloper\s+mode\b",
    r"reveal\s+(the\s+)?(system\s+)?prompt",
    r"bypass\s+(safety|guardrails|filters|restrictions)",
    r"override\s+(your\s+)?(system|safety)\s+(prompt|rules|instructions)",
]

JAILBREAK_PATTERNS = [
    r"do\s+anything\s+now\b",
    r"no\s+ethical\s+restrictions",
    r"without\s+limitations",
    r"act\s+as\s+if\s+you\s+have\s+no\s+rules",
]

# Legitimate regulatory query signals — never classify as injection.
_LEGITIMATE_REGULATORY_RE = re.compile(
    r"(?:"
    r"how\s+does|how\s+do|differ(?:s|ence)?|compare|comparison|contrast|versus|\bvs\.?\b|"
    r"what\s+(?:is|are)|which|explain|describe|list|govern|require|specify|"
    r"under\s+un\s+r|fmvss|regulation|impact\s+test|dummy|belt|restraint|anchorage"
    r")",
    re.I,
)

PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN-like"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone"),
    (r"\b\d{16}\b", "credit_card_like"),
]

UNSAFE_PATTERNS = [
    r"\b(hack|exploit|weapon)\b.*\b(vehicle|car|ecu)\b",
    r"disable\s+(airbag|restraint|safety)",
]


def is_legitimate_regulatory_query(query: str) -> bool:
    """
    Comparison, contrast, listing, and lookup questions about regulations
    must never be treated as prompt injection.
    """
    return bool(_LEGITIMATE_REGULATORY_RE.search(query))


def classify_input_state(query: str) -> InputState:
    """Classify user input: injection_blocked or answerable."""
    if is_legitimate_regulatory_query(query):
        return "answerable"
    guard = SafetyGuardrails()
    result = guard.validate_input(query)
    return "injection_blocked" if result.blocked else "answerable"


@dataclass
class GuardrailResult:
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
    input_state: InputState = "answerable"


class SafetyGuardrails:
    def __init__(self) -> None:
        self._hub_guard = None
        self._try_load_guardrails_ai()

    def _try_load_guardrails_ai(self) -> None:
        try:
            from guardrails import Guard

            self._hub_guard = Guard
            logger.info("guardrails-ai available")
        except Exception:
            logger.info("guardrails-ai hub optional — using rule-based checks")

    @staticmethod
    def _match_any(text: str, patterns: list[str]) -> str | None:
        low = text.lower()
        for p in patterns:
            if re.search(p, low, re.IGNORECASE):
                return p
        return None

    def validate_input(self, query: str) -> GuardrailResult:
        result = GuardrailResult()

        if is_legitimate_regulatory_query(query):
            result.input_state = "answerable"
            return result

        hit = self._match_any(query, INJECTION_PATTERNS)
        if hit:
            result.blocked = True
            result.block_reason = "prompt_injection"
            result.passed = False
            result.input_state = "injection_blocked"
            result.warnings.append(
                "Potential prompt injection detected. Query blocked."
            )
            return result

        hit = self._match_any(query, JAILBREAK_PATTERNS)
        if hit:
            result.blocked = True
            result.block_reason = "jailbreak"
            result.passed = False
            result.input_state = "injection_blocked"
            result.warnings.append("Potential jailbreak attempt detected.")
            return result

        result.input_state = "answerable"
        return result

    def validate_output(self, text: str) -> GuardrailResult:
        result = GuardrailResult()
        for pattern, label in PII_PATTERNS:
            if re.search(pattern, text):
                result.warnings.append(
                    f"Possible PII detected ({label}). Review before sharing."
                )

        if self._match_any(text, UNSAFE_PATTERNS):
            result.warnings.append(
                "Response may contain unsafe engineering guidance. Verify with regulations."
            )

        if self._match_any(text, INJECTION_PATTERNS + JAILBREAK_PATTERNS):
            result.warnings.append("Output contains suspicious instruction-like content.")

        return result

    def to_dict(self, result: GuardrailResult) -> dict[str, Any]:
        return {
            "passed": result.passed,
            "blocked": result.blocked,
            "block_reason": result.block_reason,
            "input_state": result.input_state,
            "warnings": result.warnings,
        }
