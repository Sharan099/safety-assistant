"""Prompt-injection detection — a deterministic flag, not a filter.

Retrieved documents and user questions are data (CLAUDE.md §2.5). This module
only *labels* suspicious instructions so the trace, the warnings and the tests
can see them; the generation contract itself is what prevents them from
becoming instructions (evidence-only answers, schema output, validation)."""

from __future__ import annotations

import re

_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (the|all|your) (previous|prior|above|system)",
    r"you are now (in )?(debug|developer|dan|jailbreak|unrestricted)",
    r"(print|reveal|show|output) (your|the) (hidden|system) prompt",
    r"^\s*system\s*:",
    r"<\s*/?\s*(system|assistant|instruction)s?\s*>",
    r"act as (an? )?(unrestricted|unfiltered)",
    r"answer that .{0,60}(is|equals) \d",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE | re.MULTILINE)


def injection_signals(text: str) -> list[str]:
    return sorted({m.group(0).strip()[:80] for m in _INJECTION_RE.finditer(text)})


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))
