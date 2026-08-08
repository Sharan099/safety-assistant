#!/usr/bin/env python3
"""Benchmark Gemini + NIM latency with thinking on vs off (Fix 17/28).

Writes ``eval/results/fix17_28_thinking_benchmark.json`` when run.

Usage:
  .venv/Scripts/python.exe scripts/benchmark_gemini_latency.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results" / "fix17_28_thinking_benchmark.json"
NVIDIA_HOST = "https://integrate.api.nvidia.com/v1"


def _one(
    client: OpenAI,
    *,
    cfg: dict,
    model: str,
    label: str,
    timeout_s: float = 25.0,
) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": 'Reply with JSON only: {"ok": true}',
                }
            ],
            temperature=0,
            max_tokens=64,
            extra_headers={"x-portkey-config": json.dumps(cfg)},
            timeout=timeout_s,
        )
        dt = (time.perf_counter() - t0) * 1000
        text = (resp.choices[0].message.content or "")[:120]
        usage = getattr(resp, "usage", None)
        usage_d = {}
        if usage is not None:
            usage_d = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return {
            "ok": True,
            "label": label,
            "ms": round(dt, 1),
            "text": text,
            "usage": usage_d,
        }
    except Exception as exc:  # noqa: BLE001
        dt = (time.perf_counter() - t0) * 1000
        return {
            "ok": False,
            "label": label,
            "ms": round(dt, 1),
            "err": str(exc)[:300],
        }


def _google_cfg(key: str, model: str, thinking: dict | None, timeout_ms: int) -> dict:
    params: dict = {"model": model}
    if thinking is not None:
        params["thinking"] = thinking
    return {
        "strategy": {"mode": "single"},
        "request_timeout": timeout_ms,
        "targets": [
            {
                "provider": "google",
                "api_key": key,
                "request_timeout": timeout_ms,
                "override_params": params,
            }
        ],
    }


def _nim_cfg(key: str, model: str, enable_thinking: bool | None, timeout_ms: int) -> dict:
    params: dict = {"model": model, "temperature": 0}
    if enable_thinking is not None:
        params["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return {
        "strategy": {"mode": "single"},
        "request_timeout": timeout_ms,
        "targets": [
            {
                "provider": "openai",
                "api_key": key,
                "custom_host": NVIDIA_HOST,
                "request_timeout": timeout_ms,
                "override_params": params,
            }
        ],
    }


def main() -> int:
    load_dotenv()
    base = (os.getenv("PORTKEY_GATEWAY_URL") or "http://localhost:8787/v1").rstrip("/")
    gkey = (os.getenv("GOOGLE_API_KEY") or "").strip()
    nkey = (os.getenv("NVIDIA_API_KEY") or "").strip()
    if not gkey and not nkey:
        raise SystemExit("Need GOOGLE_API_KEY and/or NVIDIA_API_KEY")

    client = OpenAI(api_key="x", base_url=base, timeout=30.0)
    print("gateway", base)
    rows: list[dict] = []

    if gkey:
        google_cases = [
            ("gemini-2.5-flash", None),
            ("gemini-2.5-flash", {"type": "disabled", "budget_tokens": 0}),
            ("gemini-2.5-flash-lite", None),
            ("gemini-2.5-flash-lite", {"type": "disabled", "budget_tokens": 0}),
            ("gemini-2.0-flash", None),
            ("gemini-2.0-flash-lite", None),
        ]
        for model, thinking in google_cases:
            label = (
                model
                if thinking is None
                else f"{model} thinking={thinking}"
            )
            print("---", label)
            row = _one(
                client,
                cfg=_google_cfg(gkey, model, thinking, 20000),
                model=model,
                label=label,
            )
            print(json.dumps(row, ensure_ascii=False))
            rows.append(row)

    if nkey:
        nim_model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        nim_cases = [
            (nim_model, None, "nemotron default (thinking may be ON)"),
            (nim_model, False, "nemotron enable_thinking=false"),
            (nim_model, True, "nemotron enable_thinking=false + timeout 18s"),
            ("meta/llama-3.3-70b-instruct", None, "nim llama-3.3-70b (non-reasoning)"),
        ]
        for model, thinking_flag, label in nim_cases:
            timeout = 18000 if "18s" in label else 25000
            print("---", label)
            row = _one(
                client,
                cfg=_nim_cfg(nkey, model, thinking_flag, timeout),
                model=model,
                label=label,
                timeout_s=timeout / 1000.0 + 5,
            )
            print(json.dumps(row, ensure_ascii=False))
            rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "fix": "17/28",
            "purpose": "Compare thinking-on vs thinking-off latency/token counts",
            "gateway": base,
            "recommendation": (
                "Prefer gemini-2.5-flash with thinking disabled (budget_tokens=0); "
                "flash-lite is 404 for new Google users. Prefer nemotron with "
                "enable_thinking=false over meta/llama-3.3-70b-instruct (often slower/timeout). "
                "Keep per-target request_timeout + strategy 408 failover."
            ),
        },
        "results": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
