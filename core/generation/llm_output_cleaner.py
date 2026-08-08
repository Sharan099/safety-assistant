"""Unified LLM output cleanup — display, storage, and RAGAS scoring."""

from __future__ import annotations

import re

# Qwen3 / reasoning models may emit chain-of-thought wrappers.
_THINKING_BLOCK_RE = re.compile(
    r"<\s*(?:redacted_)?think(?:ing)?\s*>.*?</\s*(?:redacted_)?think(?:ing)?\s*>",
    re.I | re.S,
)
_FENCED_THINKING_RE = re.compile(
    r"```(?:thinking|reasoning|analysis)[^\n]*\n.*?\n```",
    re.I | re.S,
)
_LEADING_THINKING_PREFIX_RE = re.compile(
    r"^(?:Thinking|Reasoning|Analysis)\s*:\s*.+?(?=\n\n|\Z)",
    re.I | re.S,
)


def clean_llm_output(text: str) -> str:
    """Strip reasoning artifacts from raw model output."""
    out = (text or "").strip()
    if not out:
        return out
    for _ in range(3):
        prev = out
        out = _THINKING_BLOCK_RE.sub("", out).strip()
        out = _FENCED_THINKING_RE.sub("", out).strip()
        out = _LEADING_THINKING_PREFIX_RE.sub("", out).strip()
        if out == prev:
            break
    return out.strip()
