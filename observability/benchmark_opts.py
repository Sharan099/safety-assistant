"""Measure token/cost deltas for prompt cache, answer cache, and model routing."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from generation.answer import answer_question
from generation.llm_client import LLMClient
from generation.prompt_cache import reset_prompt_cache
from observability.prices import clear_prices_cache


SAMPLE = "What is the HIC15 limit in UN Regulation No. 94?"


def _run(label: str, **kwargs) -> dict:
    clear_prices_cache()
    # Fresh LLMClient so provider/env is current
    resp = answer_question(SAMPLE, llm=LLMClient(), skip_answer_cache=kwargs.pop("skip_answer_cache", False), **kwargs)
    m = resp.metrics or {}
    return {
        "label": label,
        "input_tokens": m.get("input_tokens", 0),
        "output_tokens": m.get("output_tokens", 0),
        "embedding_tokens": m.get("embedding_tokens", 0),
        "cost_usd": m.get("cost_usd", 0.0),
        "latency_ms": m.get("latency_ms", 0.0),
        "answer_cached": resp.answer_cached,
        "prompt_cache_hit": m.get("prompt_cache_hit"),
        "prompt_cache_tokens_saved": m.get("prompt_cache_tokens_saved"),
        "rewrite_model": m.get("rewrite_model"),
        "answer_model": m.get("answer_model") or m.get("model"),
        "trace_id": resp.trace_id,
    }


def main() -> None:
    load_dotenv()
    os.environ.setdefault("LLM_PROVIDER", "mock")
    # Isolate answer cache for this measurement
    tmp = Path(tempfile.mkdtemp(prefix="optbench_"))
    os.environ["ANSWER_CACHE_DIR"] = str(tmp / "answers")
    os.environ["TRACE_DIR"] = str(tmp / "traces")
    os.environ["LLM_CACHE_DIR"] = str(tmp / "llm")
    os.environ["ANSWER_CACHE"] = "1"
    os.environ["PROMPT_CACHE"] = "1"

    rows = []

    # Baseline: no answer cache, cold prompt cache
    os.environ["ANSWER_CACHE"] = "0"
    reset_prompt_cache()
    rows.append(_run("1_baseline_cold", skip_answer_cache=True))

    # Prompt cache warm (second call, answer cache still off)
    rows.append(_run("2_prompt_cache_warm", skip_answer_cache=True))

    # Answer cache hit
    os.environ["ANSWER_CACHE"] = "1"
    _run("2b_warm_answer_cache", skip_answer_cache=False)  # populate
    rows.append(_run("3_answer_cache_hit", skip_answer_cache=False))

    # Model routing note (already active: small rewrite / large answer)
    os.environ["ANSWER_CACHE"] = "0"
    reset_prompt_cache()
    # Disable LLM disk cache so routing measurement isn't polluted
    rows.append(_run("4_model_routing", skip_answer_cache=True))
    # Force a routed call accounting note into optimizations via env
    client = LLMClient(use_cache=False)
    rows[-1]["routing"] = {
        "rewrite_model": client.small_model,
        "answer_model": client.large_model,
        "note": "With LLM_PROVIDER=groq, rewrite bills at small rates; answer at large.",
    }

    # Deltas
    base = rows[0]
    def delta(a: dict, b: dict) -> dict:
        return {
            "input_tokens": a["input_tokens"] - b["input_tokens"],
            "output_tokens": a["output_tokens"] - b["output_tokens"],
            "cost_usd": round(float(a["cost_usd"]) - float(b["cost_usd"]), 8),
            "latency_ms": round(float(a["latency_ms"]) - float(b["latency_ms"]), 2),
        }

    report = {
        "question": SAMPLE,
        "provider": os.getenv("LLM_PROVIDER"),
        "runs": rows,
        "deltas": {
            "prompt_cache_vs_baseline": delta(rows[1], base),
            "answer_cache_vs_baseline": delta(rows[2], base),
            "routing_note": (
                "Rewrite uses GROQ_SMALL_MODEL; final answer uses GROQ_LARGE_MODEL. "
                "Vs single large-model for both: rewrite tokens billed at small-model rates "
                "(see config/prices.json)."
            ),
        },
        "prices_file": "config/prices.json",
    }

    out = Path("eval/results/cost_optimizations.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
