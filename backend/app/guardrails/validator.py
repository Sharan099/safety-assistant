"""
Guardrails: prompt injection, jailbreak, PII leakage, unsafe content.
Uses pattern checks + optional guardrails-ai hub validators.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(the\s+)?(system|above)",
    r"you\s+are\s+now\s+(dan|evil|unrestricted)",
    r"pretend\s+you\s+are\s+not",
    r"jailbreak",
    r"developer\s+mode",
    r"reveal\s+(the\s+)?(system\s+)?prompt",
    r"bypass\s+(safety|guardrails|filters)",
]

JAILBREAK_PATTERNS = [
    r"do\s+anything\s+now",
    r"no\s+ethical\s+restrictions",
    r"without\s+limitations",
    r"act\s+as\s+if\s+you\s+have\s+no\s+rules",
]

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


@dataclass
class GuardrailResult:
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""


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
        hit = self._match_any(query, INJECTION_PATTERNS)
        if hit:
            result.blocked = True
            result.block_reason = "prompt_injection"
            result.passed = False
            result.warnings.append(
                "Potential prompt injection detected. Query blocked."
            )
            return result

        hit = self._match_any(query, JAILBREAK_PATTERNS)
        if hit:
            result.blocked = True
            result.block_reason = "jailbreak"
            result.passed = False
            result.warnings.append("Potential jailbreak attempt detected.")
            return result

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
            "warnings": result.warnings,
        }
