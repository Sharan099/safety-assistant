"""Stricter output controls when serving on the fast Groq fallback tier."""

from __future__ import annotations

import re
from collections import Counter

from config import GROQ_MODEL_FAST

from backend.app.gateway import config as cfg

_LIST_STYLE_RE = re.compile(
    r"\b(list|enumerate|each|compare|design review|requirements|clauses?|"
    r"synthesis|breakdown|step[s]?)\b",
    re.I,
)
LIST_STYLE_RE = _LIST_STYLE_RE

_FAST_MODEL_MARKERS = ("8b", "instant", "8-b")


def is_fast_groq_model(model: str | None) -> bool:
    m = (model or "").lower()
    fast = (GROQ_MODEL_FAST or "").lower()
    if fast and fast in m:
        return True
    return any(tok in m for tok in _FAST_MODEL_MARKERS)


def is_fast_fallback_tier(provider_key: str, model: str | None) -> bool:
    return provider_key == "groq" and is_fast_groq_model(model)


def effective_max_tokens(
    base_max: int,
    *,
    provider_key: str,
    model: str | None,
    prompt: str,
    fallback_used: bool,
) -> int:
    """Tighter cap on the fast tier, especially for multi-clause list prompts."""
    if not is_fast_fallback_tier(provider_key, model):
        return base_max
    if _LIST_STYLE_RE.search(prompt or ""):
        return min(base_max, int(cfg.FAST_FALLBACK_MAX_TOKENS_LIST))
    return min(base_max, int(cfg.FAST_FALLBACK_MAX_TOKENS_DEFAULT))


def _normalize_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def detect_repetition_loop(text: str, *, min_repeats: int = 3, min_len: int = 35) -> bool:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    normed = [_normalize_sentence(s) for s in sentences if len(s.strip()) >= min_len]
    if not normed:
        return False
    counts = Counter(normed)
    return max(counts.values(), default=0) >= min_repeats


def truncate_repetition(text: str, *, min_repeats: int = 3, min_len: int = 35) -> tuple[str, bool]:
    """
    Return (possibly truncated text, was_truncated).
    Stops at the third repeat of the same sentence template.
    """
    if not text:
        return text, False
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    kept: list[str] = []
    seen: Counter[str] = Counter()
    truncated = False
    for part in parts:
        norm = _normalize_sentence(part)
        if len(norm) >= min_len:
            seen[norm] += 1
            if seen[norm] >= min_repeats:
                truncated = True
                break
        kept.append(part)
    out = " ".join(kept).strip()
    if truncated and out and not out.endswith((".", "!", "?")):
        out += "…"
    return out, truncated


def sanitize_fast_model_output(
    answer: str,
    *,
    provider_key: str,
    model: str | None,
) -> tuple[str, dict]:
    """Post-generation guard for fast-tier loops. Returns (text, metadata)."""
    meta: dict = {"fast_fallback_safeguard": False}
    if not is_fast_fallback_tier(provider_key, model):
        return answer, meta
    meta["fast_fallback_safeguard"] = True
    if not detect_repetition_loop(answer):
        return answer, meta
    cleaned, truncated = truncate_repetition(answer)
    meta["repetition_truncated"] = truncated
    if truncated:
        cleaned = (
            f"{cleaned}\n\n"
            "(Response shortened — fast-tier safeguard detected repeated text.)"
        )
    return cleaned, meta
