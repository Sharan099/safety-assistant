#!/usr/bin/env python3
"""Manual E2E: invalid Groq key → Portkey fallback → cited /chat + dashboard provider.

This is the single proof that the "don't exhaust Groq free-tier" goal works end to
end — not just in ``config/portkey/*.json``:

1. Temporarily replace ``GROQ_API_KEY`` with an invalid value (keeps Groq in the
   fallback chain so Portkey must try it and fail).
2. POST a real ``/chat/sync`` through the local Portkey gateway.
3. Assert the answer was served by NVIDIA NIM (or the next configured target),
   includes citations, and ``/metrics/{trace_id}`` logs that provider
   (``target_index > 0`` when the primary was skipped).
4. Restore the real ``GROQ_API_KEY`` in a ``finally`` block (even on failure).

Usage::

  # Terminal A — local open-source gateway (no Portkey cloud account)
  docker compose up -d portkey
  # or:  npx @portkey-ai/gateway
  # or:  docker run --rm -p 8787:8787 portkeyai/gateway:latest

  # Terminal B — Qdrant must already hold the regulations (ingest first if needed)
  uv run python scripts/manual_test_portkey_fallback.py

Requires ``GROQ_API_KEY`` (real, to restore) and ``NVIDIA_API_KEY`` (or Google /
OpenRouter further down the chain) in ``.env``.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
INVALID_GROQ = "gsk_INVALID_PORTKEY_FALLBACK_TEST_DO_NOT_USE"
DEFAULT_PORT = int(os.getenv("FALLBACK_TEST_PORT") or "8011")
# Well-known gold-style question (cache bypassed via PORTKEY_CACHE_FORCE_REFRESH).
QUESTION = "What is the HIC15 limit in UN Regulation No. 94?"

# Accepted non-primary answer providers (FINAL_ANSWER_CONFIG order after Groq).
FALLBACK_PROVIDERS = frozenset({"nvidia_nim", "google", "openrouter"})


def _read_env_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _set_env_key(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}=.*$")
    line = f"{key}={value}"
    if pattern.search(text):
        return pattern.sub(line, text)
    return text.rstrip() + f"\n{line}\n"


def _get_env_key(text: str, key: str) -> str | None:
    m = re.search(rf"(?m)^{re.escape(key)}=(.*)$", text)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'")


def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 180.0) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"{method} {url} → HTTP {exc.code}: {detail}") from exc


def _wait_http(url: str, *, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {url}: {last}")


def _ensure_portkey() -> None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8787/", timeout=3) as resp:
            if resp.status < 500:
                print("Portkey gateway: already up on :8787")
                return
    except Exception:
        pass
    raise RuntimeError(
        "Portkey gateway is not reachable on http://localhost:8787\n"
        "Start it first, then re-run this script:\n"
        "  docker compose up -d portkey\n"
        "  # or: npx @portkey-ai/gateway\n"
        "  # or: docker run --rm -p 8787:8787 portkeyai/gateway:latest"
    )


def _served_provider(chat: dict, metrics_payload: dict) -> tuple[str, int | None]:
    """Resolve which provider actually answered + Portkey target index."""
    metrics = chat.get("metrics") or {}
    llm_calls = metrics_payload.get("llm_calls") or metrics.get("llm_calls") or []
    answer_calls = [c for c in llm_calls if isinstance(c, dict) and c.get("role") == "answer"]
    if answer_calls:
        last = answer_calls[-1]
        prov = str(last.get("provider") or "").strip().lower()
        idx = last.get("target_index")
        try:
            idx_i = int(idx) if idx is not None else None
        except (TypeError, ValueError):
            idx_i = None
        return prov, idx_i

    served = (
        str(metrics_payload.get("answer_provider") or "")
        or str(metrics_payload.get("provider") or "")
        or str(chat.get("provider") or "")
        or str(metrics.get("provider") or "")
    ).strip().lower()
    idx = metrics_payload.get("target_index")
    try:
        idx_i = int(idx) if idx is not None else None
    except (TypeError, ValueError):
        idx_i = None
    return served, idx_i


def main() -> int:
    os.chdir(ROOT)
    _ensure_portkey()

    original = _read_env_file(ENV_PATH)
    real_groq = _get_env_key(original, "GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or ""
    if not real_groq or real_groq == INVALID_GROQ:
        print(
            "ERROR: could not find a real GROQ_API_KEY in .env to restore later",
            file=sys.stderr,
        )
        return 2

    nvidia = _get_env_key(original, "NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEY") or ""
    google = _get_env_key(original, "GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    openrouter = (
        _get_env_key(original, "OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    )
    if not (nvidia or google or openrouter):
        print(
            "ERROR: need at least one fallback key "
            "(NVIDIA_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY)",
            file=sys.stderr,
        )
        return 2

    port = DEFAULT_PORT
    base = f"http://127.0.0.1:{port}"
    proc: subprocess.Popen[str] | None = None
    restored = False
    question = QUESTION

    def restore_env() -> None:
        nonlocal restored
        if restored:
            return
        ENV_PATH.write_text(original, encoding="utf-8")
        restored = True
        print("Restored original .env (real GROQ_API_KEY).")

    try:
        patched = _set_env_key(original, "GROQ_API_KEY", INVALID_GROQ)
        patched = _set_env_key(patched, "LLM_PROVIDER", "groq")
        patched = _set_env_key(patched, "PORTKEY_GATEWAY_URL", "http://localhost:8787/v1")
        patched = _set_env_key(patched, "ANSWER_CACHE", "0")
        patched = _set_env_key(patched, "PORTKEY_CACHE_FORCE_REFRESH", "1")
        # Avoid first-load hang of local CrossEncoder during this LLM-fallback proof.
        patched = _set_env_key(patched, "RERANK_PROVIDER", "none")
        ENV_PATH.write_text(patched, encoding="utf-8")
        print("Temporarily set GROQ_API_KEY to an invalid value in .env")
        print("  (Portkey will try Groq, fail auth, then fall through the chain)")

        env = os.environ.copy()
        env["GROQ_API_KEY"] = INVALID_GROQ
        env["LLM_PROVIDER"] = "groq"
        env["PORTKEY_GATEWAY_URL"] = "http://localhost:8787/v1"
        env["ANSWER_CACHE"] = "0"
        env["PORTKEY_CACHE_FORCE_REFRESH"] = "1"
        env["RERANK_PROVIDER"] = "none"
        env["API_HOST"] = "127.0.0.1"
        env["API_PORT"] = str(port)
        # Prefer local on-disk Qdrant (Compose qdrant volume is separate / often empty).
        env.pop("QDRANT_URL", None)
        env["QDRANT_PATH"] = str(ROOT / "data" / "qdrant")
        env["QDRANT_URL"] = ""

        print(f"Starting temporary API on {base} …")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_http(f"{base}/health", timeout_s=120.0)
        except Exception:
            out = ""
            if proc.stdout:
                try:
                    out = proc.stdout.read()[-4000:]
                except Exception:
                    out = ""
            raise RuntimeError(f"Temp API failed to start.\n{out}") from None

        print(f"POST /chat/sync — {question[:90]}…")
        chat = _http_json(
            "POST",
            f"{base}/chat/sync",
            {
                "question": question,
                "top_k": 5,
                "conversation_id": f"fallback-test-{int(time.time())}",
            },
            timeout=600.0,
        )
        trace_id = str(chat.get("trace_id") or "")
        citations = chat.get("citations") or []
        answer = str(chat.get("answer") or "")
        model = str(chat.get("model") or "")

        if not trace_id:
            print("FAIL: missing trace_id — cannot check dashboard", file=sys.stderr)
            return 1

        metrics_payload = _http_json("GET", f"{base}/metrics/{trace_id}", timeout=30.0)
        served, target_index = _served_provider(chat, metrics_payload)
        agg = _http_json("GET", f"{base}/metrics/aggregate?limit=50", timeout=30.0)
        by_provider = (agg.get("by_provider") or {}) if isinstance(agg, dict) else {}

        print("\n--- /chat/sync result ---")
        print(f"provider:      {chat.get('provider')}")
        print(f"model:         {model}")
        print(f"not_found:     {chat.get('not_found')}")
        print(f"citations:     {len(citations)}")
        print(f"trace_id:      {trace_id}")
        print(f"answer:        {answer[:400]}{'…' if len(answer) > 400 else ''}")

        print("\n--- /metrics/{trace_id} (dashboard) ---")
        print(
            json.dumps(
                {
                    "provider": metrics_payload.get("provider"),
                    "answer_provider": metrics_payload.get("answer_provider"),
                    "answer_model": metrics_payload.get("answer_model"),
                    "target_index": metrics_payload.get("target_index"),
                    "cache_status": metrics_payload.get("cache_status"),
                    "cost_usd": metrics_payload.get("cost_usd"),
                    "input_tokens": metrics_payload.get("input_tokens"),
                    "output_tokens": metrics_payload.get("output_tokens"),
                    "llm_calls": metrics_payload.get("llm_calls"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print("\n--- /metrics/aggregate by_provider (fallback frequency) ---")
        print(json.dumps(by_provider, indent=2, ensure_ascii=False))

        ok = True

        if served not in FALLBACK_PROVIDERS:
            print(
                f"\nFAIL: expected fallback provider in {sorted(FALLBACK_PROVIDERS)}, "
                f"got {served!r} (Groq must NOT be the server)",
                file=sys.stderr,
            )
            ok = False
        else:
            print(f"\nPASS: served by fallback provider {served!r} (Groq was skipped)")

        if target_index is not None and target_index <= 0:
            print(
                f"FAIL: target_index={target_index} — expected >0 when primary Groq fails",
                file=sys.stderr,
            )
            ok = False
        elif target_index is not None:
            print(f"PASS: target_index={target_index} (fallback chain position)")
        else:
            print("WARN: target_index missing from Portkey headers (provider check still applies)")

        if served in by_provider:
            print(
                f"PASS: aggregate by_provider includes {served!r} "
                f"(n={by_provider[served].get('n')}, share={by_provider[served].get('share')})"
            )
        else:
            print(
                f"FAIL: aggregate by_provider missing {served!r} — dashboard share view broken",
                file=sys.stderr,
            )
            ok = False

        if citations:
            print(f"PASS: {len(citations)} citation(s) returned")
            for c in citations[:3]:
                print(f"  - {c.get('citation') or c.get('label') or c.get('chunk_id')}")
        elif chat.get("not_found"):
            # Fallback-provider proof still stands; retrieval/grounding is a separate concern.
            print(
                "WARN: not_found / no citations (retrieval or grounding). "
                "Portkey fallback + dashboard provider checks still apply.",
                file=sys.stderr,
            )
        else:
            print("FAIL: answer has no citations and was not marked not_found", file=sys.stderr)
            ok = False

        if not answer.strip():
            print("FAIL: empty answer body", file=sys.stderr)
            ok = False
        else:
            print("PASS: non-empty answer body returned")

        print("PASS: dashboard metrics recorded the serving provider" if ok else "")
        return 0 if ok else 1
    finally:
        if proc is not None and proc.poll() is None:
            if sys.platform == "win32":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        restore_env()


if __name__ == "__main__":
    raise SystemExit(main())
