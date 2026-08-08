#!/usr/bin/env python3
"""Write FreeLLMAPI declarative config from this project's .env (no secrets logged).

Reuses Groq / Google / NVIDIA / OpenRouter keys already in .env.
Seeds anon/keyless overflow: OVH, AI Horde; LLM7 via documented anonymous token "unused".
Pollinations is skipped unless POLLINATIONS_API_KEY is set (upstream no longer keyless).
Never adds Cohere (ToS forbids personal/household use per FreeLLMAPI review table).
Does not add GitHub Models — leave unkeyed; eval-only if enabled later.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
OUT_PATH = ROOT / "data" / "freellmapi" / "runtime.config.json"


def _load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def main() -> int:
    env = {**_load_dotenv(ENV_PATH), **os.environ}
    keys: list[dict] = [
        # Keyless / anon overflow (zero setup)
        {"platform": "ovh", "label": "anon"},
        {"platform": "aihorde", "label": "anon"},
        # LLM7 documents api_key="unused" for anonymous access
        {"platform": "llm7", "key": "unused", "label": "anon"},
    ]

    # Pollinations: FreeLLMAPI #573 removed keyless; only seed if a key exists.
    pollinations = (env.get("POLLINATIONS_API_KEY") or "").strip()
    if pollinations:
        keys.append({"platform": "pollinations", "key": pollinations, "label": "main"})

    keyed = [
        ("groq", "GROQ_API_KEY"),
        # Google AI Studio / Gemini — credential reused; ToS: eval-only via FreeLLMAPI path
        ("google", "GOOGLE_API_KEY"),
        ("nvidia", "NVIDIA_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ]
    missing = []
    for platform, env_name in keyed:
        val = (env.get(env_name) or "").strip()
        if not val:
            missing.append(env_name)
            continue
        keys.append({"platform": platform, "key": val, "label": "main", "enabled": True})

    if missing:
        print(f"warning: missing env for FreeLLMAPI seed: {', '.join(missing)}", file=sys.stderr)

    # Explicitly do NOT include: cohere, github (GitHub Models = eval-only, no key here).
    config = {
        "keys": keys,
        "routing": {"strategy": "balanced"},
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    # Never print key material — only platform labels and counts.
    platforms = [k["platform"] for k in keys]
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} with {len(keys)} key entries: {', '.join(platforms)}")
    if not pollinations:
        print(
            "note: Pollinations not seeded (needs POLLINATIONS_API_KEY; FreeLLMAPI no longer keyless).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
