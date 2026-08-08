"""Pinned eval judge for RAGAS / DeepEval / DeepTeam scoring.

Loads ``config/portkey/eval_judge_pinned.json`` — a **single** provider/model,
not Portkey's production ``judge.json`` fallback chain and not a multi-provider
overflow ladder. Scores stay comparable run-over-run regardless of which
provider answered the SUT question.

Model pin (first match wins):
  ``EVAL_JUDGE_MODEL`` → ``RAGAS_JUDGE_MODEL`` → ``SECURITY_JUDGE_MODEL`` →
  pinned JSON primary (default ``gemini-2.5-flash``).

Optional provider pin: ``EVAL_JUDGE_PROVIDER`` (default from JSON, usually
``google``).

Production paths (FINAL_ANSWER / QUERY_REWRITE / ``judge.json``) are unchanged;
``generation/llm_client.py`` has no reference to this config or FreeLLMAPI.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

from generation.llm_client import LLMClient, LLMRole, load_portkey_config

# Disk basename under config/portkey/ (no .json).
EVAL_JUDGE_PINNED_CONFIG_NAME = "eval_judge_pinned"
# Back-compat alias — install_* still imported under the old overflow name.
EVAL_JUDGE_OVERFLOW_CONFIG_NAME = EVAL_JUDGE_PINNED_CONFIG_NAME

DEFAULT_PINNED_JUDGE_MODEL = "gemini-2.5-flash"
DEFAULT_PINNED_JUDGE_PROVIDER = "google"


def _env_judge_model() -> str:
    return (
        (os.getenv("EVAL_JUDGE_MODEL") or "").strip()
        or (os.getenv("RAGAS_JUDGE_MODEL") or "").strip()
        or (os.getenv("SECURITY_JUDGE_MODEL") or "").strip()
    )


def _env_judge_provider() -> str:
    return (os.getenv("EVAL_JUDGE_PROVIDER") or "").strip().lower()


def EVAL_JUDGE_OVERFLOW_CONFIG() -> dict[str, Any]:
    """Resolved Portkey config for the pinned eval judge (keys from env)."""
    return build_pinned_eval_judge_config()


def build_pinned_eval_judge_config() -> dict[str, Any]:
    """Single-target Portkey config — no production / multi-provider fallback."""
    base = load_portkey_config(EVAL_JUDGE_PINNED_CONFIG_NAME)
    cfg = json.loads(json.dumps(base))  # deep copy
    targets = cfg.get("targets") if isinstance(cfg.get("targets"), list) else []
    if not targets:
        raise RuntimeError(
            "EVAL_JUDGE_PINNED has no usable targets — set GOOGLE_API_KEY "
            "(or EVAL_JUDGE_PROVIDER + matching API key)"
        )

    # Keep exactly one target so Portkey cannot fall through to another model.
    target = dict(targets[0])
    provider = _env_judge_provider() or str(target.get("provider") or DEFAULT_PINNED_JUDGE_PROVIDER)
    target["provider"] = provider

    model = _env_judge_model()
    params = dict(target.get("override_params") or {})
    if model:
        params["model"] = model
    elif not params.get("model"):
        params["model"] = DEFAULT_PINNED_JUDGE_MODEL
    # Strip provider-specific extras when forcing a non-google pin.
    if provider != "google":
        params.pop("thinking", None)
    if provider == "groq" and not target.get("api_key"):
        target["api_key"] = "${GROQ_API_KEY}"
    if provider == "google" and not target.get("api_key"):
        target["api_key"] = "${GOOGLE_API_KEY}"
    target["override_params"] = params

    cfg["targets"] = [target]
    # Single target only — Portkey may still advertise fallback strategy, but
    # there is no second model to fall through to (scores stay model-pinned).
    cfg["strategy"] = {
        "mode": "fallback",
        "on_status_codes": [404, 408, 413, 429, 500, 502, 503, 504],
    }
    return cfg


def eval_judge_overflow_primary_model() -> str:
    cfg = build_pinned_eval_judge_config()
    for t in cfg.get("targets") or []:
        if not isinstance(t, dict):
            continue
        params = t.get("override_params") if isinstance(t.get("override_params"), dict) else {}
        model = params.get("model")
        if model:
            return str(model)
    return _env_judge_model() or DEFAULT_PINNED_JUDGE_MODEL


def install_eval_judge_overflow(client: LLMClient) -> dict[str, Any]:
    """Bind ``client.judge()`` to the pinned eval judge for this instance.

    Only ``LLMRole.JUDGE`` is redirected; rewrite/answer on the same client still
    use QUERY_REWRITE / FINAL_ANSWER. Safe to call once per scoring client.
    """
    pinned = build_pinned_eval_judge_config()
    if not pinned.get("targets"):
        raise RuntimeError(
            "Pinned eval judge has no usable targets — set EVAL_JUDGE_MODEL / "
            "GOOGLE_API_KEY (or EVAL_JUDGE_PROVIDER + key)"
        )
    if getattr(client, "_eval_judge_overflow_bound", False):
        return pinned

    original = client.portkey_config_for

    def _portkey_config_for(role: LLMRole | str) -> dict[str, Any]:
        if LLMRole(role) is LLMRole.JUDGE:
            # Fresh copy each call; do not let LLMClient rewrite first-target model.
            return json.loads(json.dumps(pinned))
        return original(role)

    client.portkey_config_for = _portkey_config_for  # type: ignore[method-assign]
    client._eval_judge_overflow_bound = True  # type: ignore[attr-defined]
    return pinned


@contextmanager
def bind_eval_judge_overflow(client: LLMClient) -> Iterator[dict[str, Any]]:
    """Context-manager form of :func:`install_eval_judge_overflow` (restores on exit)."""
    pinned = build_pinned_eval_judge_config()
    if not pinned.get("targets"):
        raise RuntimeError(
            "Pinned eval judge has no usable targets — set EVAL_JUDGE_MODEL / "
            "GOOGLE_API_KEY (or EVAL_JUDGE_PROVIDER + key)"
        )
    original = client.portkey_config_for
    was_bound = getattr(client, "_eval_judge_overflow_bound", False)

    def _portkey_config_for(role: LLMRole | str) -> dict[str, Any]:
        if LLMRole(role) is LLMRole.JUDGE:
            return json.loads(json.dumps(pinned))
        return original(role)

    client.portkey_config_for = _portkey_config_for  # type: ignore[method-assign]
    client._eval_judge_overflow_bound = True  # type: ignore[attr-defined]
    try:
        yield pinned
    finally:
        client.portkey_config_for = original  # type: ignore[method-assign]
        client._eval_judge_overflow_bound = was_bound  # type: ignore[attr-defined]
