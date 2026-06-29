"""
Gateway-side prompt trimming and compression.

Runs inside the gateway so retrieval/graph code stays unchanged. Parses the
workflow prompt shape (RETRIEVED CONTEXT + [S#] passages) and enforces passage
count and token budgets before any provider call.
"""

from __future__ import annotations

import os
import re

from backend.app.gateway.model_registry import ModelSpec

_PASSAGE_MARKER_RE = re.compile(r"(?=\[S\d+\])")
_CONTEXT_HEADER = "RETRIEVED CONTEXT"
_QUESTION_HEADER = "QUESTION:"


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


GATEWAY_MAX_PASSAGES = _i("GATEWAY_MAX_PASSAGES", 6)
GATEWAY_CONTEXT_TOKEN_BUDGET = _i("GATEWAY_CONTEXT_TOKEN_BUDGET", 3_500)
GATEWAY_OUTPUT_TOKEN_RESERVE = _i("GATEWAY_OUTPUT_TOKEN_RESERVE", 1_024)


def estimate_tokens(text: str) -> int:
    return max(1, int(len((text or "").split()) * 1.33))


def trim_prompt_for_gateway(prompt: str) -> str:
    """Cap passage count and context token budget before routing."""
    if not prompt or _CONTEXT_HEADER not in prompt:
        return _truncate_text(prompt, GATEWAY_CONTEXT_TOKEN_BUDGET)

    before, rest = prompt.split(_CONTEXT_HEADER, 1)
    header = _CONTEXT_HEADER + rest.split(_QUESTION_HEADER, 1)[0]
    question_part = ""
    if _QUESTION_HEADER in rest:
        question_part = _QUESTION_HEADER + rest.split(_QUESTION_HEADER, 1)[1]

    passages = _split_passages(header)
    if len(passages) > GATEWAY_MAX_PASSAGES:
        passages = passages[:GATEWAY_MAX_PASSAGES]

    context_body = "\n\n".join(passages).strip()
    context_body = _truncate_text(context_body, GATEWAY_CONTEXT_TOKEN_BUDGET)
    return f"{before}{_CONTEXT_HEADER}\n{context_body}\n\n{question_part}".strip()


def fit_messages_for_model(
    messages: list[dict[str, str]],
    spec: ModelSpec,
    *,
    max_output_tokens: int,
) -> list[dict[str, str]]:
    """Pre-flight: ensure messages fit the model effective request limit."""
    reserve = max_output_tokens + GATEWAY_OUTPUT_TOKEN_RESERVE
    budget = max(512, spec.effective_request_tokens - reserve)
    out: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user" and _CONTEXT_HEADER in content:
            content = trim_prompt_for_gateway(content)
        content = _truncate_text(content, budget)
        budget -= estimate_tokens(content)
        out.append({"role": role, "content": content})
    return out


def compress_messages(
    messages: list[dict[str, str]],
    spec: ModelSpec,
    *,
    level: int,
    max_output_tokens: int,
) -> list[dict[str, str]]:
    """
    Progressive compression for 413 retries on the SAME model.
    level 1: drop last passage; level 2+: shorten passage text.
    """
    out: list[dict[str, str]] = []
    for msg in messages:
        content = msg.get("content", "")
        if msg.get("role") == "user" and _CONTEXT_HEADER in content:
            content = _compress_user_prompt(content, level=level)
        out.append({"role": msg.get("role", "user"), "content": content})
    return fit_messages_for_model(out, spec, max_output_tokens=max_output_tokens)


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def _split_passages(context_block: str) -> list[str]:
    parts = _PASSAGE_MARKER_RE.split(context_block)
    return [p.strip() for p in parts if p.strip() and "[S" in p]


def _compress_user_prompt(prompt: str, *, level: int) -> str:
    if _CONTEXT_HEADER not in prompt:
        return _truncate_text(prompt, max(400, GATEWAY_CONTEXT_TOKEN_BUDGET // (level + 1)))

    before, rest = prompt.split(_CONTEXT_HEADER, 1)
    question_part = ""
    ctx = rest
    if _QUESTION_HEADER in rest:
        ctx, question_part = rest.split(_QUESTION_HEADER, 1)
        question_part = _QUESTION_HEADER + question_part

    passages = _split_passages(ctx)
    if not passages:
        return prompt

    drop = min(level, max(0, len(passages) - 1))
    if drop:
        passages = passages[: max(1, len(passages) - drop)]

    shrink = max(0, level - 1)
    if shrink:
        factor = max(0.35, 1.0 - 0.25 * shrink)
        passages = [_truncate_text(p, int(len(p) * factor)) for p in passages]

    ctx_body = "\n\n".join(passages)
    budget = max(300, GATEWAY_CONTEXT_TOKEN_BUDGET // (level + 1))
    ctx_body = _truncate_text(ctx_body, budget)
    return f"{before}{_CONTEXT_HEADER}\n{ctx_body}\n\n{question_part}".strip()


def _truncate_text(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    words = (text or "").split()
    keep = max(1, int(max_tokens / 1.33))
    return " ".join(words[:keep]) + "\n…[truncated for token budget]"
