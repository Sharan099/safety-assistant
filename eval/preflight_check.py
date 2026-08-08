"""Preflight check: ping every Portkey fallback target *directly* before a full eval run.

Sends one trivial prompt ("Reply with the single word OK.") to each configured
target in ``config/portkey/final_answer.json`` (Groq, NVIDIA NIM, Google Gemini,
OpenRouter) individually — bypassing Portkey's fallback ``strategy`` entirely by
sending a single-target config per call — with a short (10s default) timeout.

Why this matters: if a provider is down/misconfigured, the fallback chain will
silently route around it during the full run, pushing more load onto the
remaining providers and changing the run's cost/latency profile (and possibly
its pass/fail results, since judge/SUT model choice affects scoring). Catching
that *before* the golden set runs — for the price of at most one call per
configured target (typically <= 4 total) — is much cheaper than discovering it
100 cases into a multi-hour run.

Usage::

    python -m eval.preflight_check
    python -m eval.preflight_check --yes   # proceed even if a provider fails
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from generation.llm_client import (
    LLMRole,
    apply_no_think_system_prefix,
    config_for_role,
    gateway_base_url,
    is_reasoning_nim_model,
)

logger = logging.getLogger(__name__)

PING_PROMPT = "Reply with the single word OK."
DEFAULT_TIMEOUT_S = 10.0
_YES_VALUES = {"1", "true", "yes", "y"}


@dataclass
class PreflightResult:
    provider: str
    model: str
    status: str  # "OK" | "TIMEOUT" | "ERROR"
    latency_s: float | None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "OK"


def _target_provider_label(target: dict[str, Any]) -> str:
    meta = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
    return str(meta.get("logical_provider") or target.get("provider") or "unknown")


def _target_model_label(target: dict[str, Any]) -> str:
    params = target.get("override_params") if isinstance(target.get("override_params"), dict) else {}
    return str((params or {}).get("model") or "unknown")


def _single_target_config(target: dict[str, Any]) -> dict[str, Any]:
    """A Portkey config with only ONE target.

    The Portkey OSS gateway requires either a bare ``provider``/``api_key`` pair
    or a ``strategy``+``targets`` config, so we keep the ``fallback`` strategy
    shape — but with a single target there is nothing left to fall back *to*,
    which is what makes this a direct, non-fallback ping.
    """
    return {
        "strategy": {"mode": "fallback"},
        "targets": [copy.deepcopy(target)],
    }


def ping_target(
    target: dict[str, Any],
    *,
    gateway_url: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> PreflightResult:
    """Send one trivial completion straight to ``target``, no fallback chain."""
    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

    provider = _target_provider_label(target)
    model = _target_model_label(target)
    cfg = _single_target_config(target)

    client = OpenAI(
        api_key="not-needed",  # per-target keys live in x-portkey-config
        base_url=gateway_url,
        timeout=timeout_s,
        default_headers={"User-Agent": "passive-safety-rag/preflight"},
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": PING_PROMPT}]
    if is_reasoning_nim_model(model):
        # Reasoning NIM models default to long chain-of-thought; force it off so
        # the trivial ping doesn't burn its 8-token budget on <think> output.
        messages = apply_no_think_system_prefix(messages, enabled=True)

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=32,
            extra_headers={"x-portkey-config": json.dumps(cfg)},
        )
    except APITimeoutError as exc:
        return PreflightResult(provider, model, "TIMEOUT", None, str(exc)[:200])
    except (APIConnectionError, APIStatusError) as exc:
        latency = time.perf_counter() - t0
        return PreflightResult(provider, model, "ERROR", round(latency, 2), str(exc)[:200])
    except Exception as exc:  # noqa: BLE001 — surface anything unexpected as a failure, not a crash
        latency = time.perf_counter() - t0
        status = "TIMEOUT" if "timeout" in type(exc).__name__.lower() else "ERROR"
        return PreflightResult(provider, model, status, None if status == "TIMEOUT" else round(latency, 2), str(exc)[:200])

    latency = time.perf_counter() - t0
    text = ""
    if resp.choices:
        message = resp.choices[0].message
        text = (getattr(message, "content", None) or "").strip()
        if not text:
            # Reasoning models (e.g. NIM Nemotron) may put output in `reasoning`.
            reasoning = getattr(message, "reasoning", None)
            if isinstance(reasoning, str):
                text = reasoning.strip()
    if not text:
        return PreflightResult(provider, model, "ERROR", round(latency, 2), "empty response")
    return PreflightResult(provider, model, "OK", round(latency, 2), text[:40])


def run_preflight(*, timeout_s: float = DEFAULT_TIMEOUT_S) -> list[PreflightResult]:
    """Ping every configured target in the final-answer fallback chain directly.

    Only targets with an API key configured are included (unconfigured targets
    are already dropped by ``load_portkey_config``), so this costs at most one
    call per *live* provider — typically <= 4 (Groq, NVIDIA NIM, Google, OpenRouter).
    """
    load_dotenv(override=False)
    gateway_url = gateway_base_url()
    cfg = config_for_role(LLMRole.ANSWER)
    targets = [t for t in (cfg.get("targets") or []) if isinstance(t, dict)]
    if not targets:
        raise RuntimeError(
            "No configured Portkey targets found (check GROQ_API_KEY / NVIDIA_API_KEY / "
            "GOOGLE_API_KEY / OPENROUTER_API_KEY in .env and config/portkey/final_answer.json)"
        )
    return [ping_target(t, gateway_url=gateway_url, timeout_s=timeout_s) for t in targets]


def format_table(results: list[PreflightResult]) -> str:
    col_provider, col_model, col_status = 16, 30, 8
    header = f"{'provider':<{col_provider}} {'model':<{col_model}} {'status':<{col_status}} latency"
    lines = [header, "-" * len(header)]
    for r in results:
        latency_str = f"{r.latency_s:.1f}s" if r.latency_s is not None else "-"
        lines.append(
            f"{r.provider:<{col_provider}} {r.model:<{col_model}} {r.status:<{col_status}} {latency_str}"
        )
    return "\n".join(lines)


def all_ok(results: list[PreflightResult]) -> bool:
    return all(r.ok for r in results)


def confirm_or_abort(results: list[PreflightResult], *, assume_yes: bool = False) -> bool:
    """Return True if it's OK to proceed with the full eval run."""
    if all_ok(results):
        return True

    failed = [r for r in results if not r.ok]
    print("\nWARNING: preflight check found unhealthy provider(s):", file=sys.stderr)
    for r in failed:
        detail = f" — {r.detail}" if r.detail else ""
        print(f"  - {r.provider} ({r.model}): {r.status}{detail}", file=sys.stderr)
    print(
        "A failed provider means the fallback chain will push more load onto the "
        "remaining providers during the full run, changing its cost/latency profile "
        "(and possibly SUT/judge model choice, which can affect scores).",
        file=sys.stderr,
    )

    if assume_yes:
        print("Proceeding anyway (--yes / EVAL_ASSUME_YES set).", file=sys.stderr)
        return True
    if not sys.stdin.isatty():
        print(
            "stdin is not a TTY (non-interactive run) — pass --yes to proceed despite "
            "the failed provider(s) above, or fix the provider and re-run.",
            file=sys.stderr,
        )
        return False
    try:
        reply = input("Continue with the full eval run anyway? [y/N] ").strip().lower()
    except EOFError:
        reply = ""
    return reply in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ping every Portkey fallback target directly (no fallback logic) before a full eval run"
    )
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="Per-provider timeout in seconds")
    p.add_argument("--yes", "-y", action="store_true", help="Proceed even if a provider fails, without prompting")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    print(f"Preflight: pinging each provider directly with {args.timeout:.0f}s timeout...\n")
    results = run_preflight(timeout_s=args.timeout)
    print(format_table(results))

    assume_yes = args.yes or (os.getenv("EVAL_ASSUME_YES") or "").strip().lower() in _YES_VALUES
    if confirm_or_abort(results, assume_yes=assume_yes):
        if all_ok(results):
            print("\nAll providers healthy.")
        return 0
    print("Aborted.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
