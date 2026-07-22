"""
RAGAS evaluation — answerable vs abstention tracks, Groq judge only.

Requires a running API (default http://127.0.0.1:8002/api/v1) and GROQ_API_KEY.

    python -m evaluation.run
    python -m evaluation.run --api=http://127.0.0.1:8002/api/v1
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
import httpx

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

from evaluation.scoring import (  # noqa: E402
    ABSTENTION,
    ANSWERABLE,
    RagasScoringError,
    overall_metrics,
    run_ragas,
    track_metrics,
)

CASES_PATH = ROOT / "cases.json"
RESULTS_PATH = ROOT / "results.json"
DEFAULT_API = os.getenv("EVAL_API_URL", "http://127.0.0.1:8002/api/v1")
TOP_K = int(os.getenv("EVAL_TOP_K", "8"))
DELAY_SEC = float(os.getenv("EVAL_INTER_REQUEST_DELAY_SEC", "12"))


def load_cases() -> list[dict[str, Any]]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if len(cases) != 10:
        raise ValueError(f"Expected exactly 10 eval cases, found {len(cases)}")
    for case in cases:
        track = case.get("track") or ANSWERABLE
        if track not in (ANSWERABLE, ABSTENTION):
            raise ValueError(f"Case {case.get('id')}: track must be answerable|abstention")
        case["track"] = track
    return cases


def _chunks(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "regulation_code": s.get("regulation_code"),
            "section": s.get("section"),
            "snippet": (s.get("chunk_text") or s.get("snippet") or "").strip(),
        }
        for s in sources
    ]


def collect_case(client: httpx.Client, case: dict[str, Any], api_base: str) -> dict[str, Any]:
    query = case["query"]
    search = client.get(f"{api_base}/search", params={"q": query, "top_k": TOP_K}, timeout=600.0)
    search.raise_for_status()
    sources = search.json().get("sources") or []

    chat = client.post(f"{api_base}/chat", json={"query": query, "top_k": TOP_K}, timeout=600.0)
    chat.raise_for_status()
    data = chat.json()

    return {
        "id": case["id"],
        "track": case["track"],
        "query": query,
        "ground_truth": case.get("ground_truth", ""),
        "answer": data.get("answer", ""),
        "retrieved_chunks": _chunks(sources),
        "retrieved_context": "\n\n".join(
            (s.get("chunk_text") or s.get("snippet") or "") for s in sources
        ),
        "citations": data.get("citations") or [],
        "route": data.get("route", "regulatory"),
        "gateway": data.get("gateway") or {},
    }


def collect_all(cases: list[dict[str, Any]], api_base: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for i, case in enumerate(cases):
            if i:
                time.sleep(DELAY_SEC)
            rec = collect_case(client, case, api_base)
            logger.info("Collected {} ({})", rec["id"], rec["track"])
            records.append(rec)
    return records


def run(api_base: str | None = None) -> dict[str, Any]:
    base = (api_base or DEFAULT_API).rstrip("/")
    cases = load_cases()
    logger.info("Collecting {} answers from {}", len(cases), base)
    records = collect_all(cases, base)
    records = run_ragas(records)
    overall = overall_metrics(records)
    tracks = track_metrics(records)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": "ragas+abstention_binary",
        "judge": "groq",
        "n_questions": len(records),
        "api_base": base,
        "tracks": tracks,
        "overall_metrics": overall,
        "cases": [
            {
                "id": r["id"],
                "track": r["track"],
                "query": r["query"],
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
                "metrics": r["metrics"],
                "metrics_source": r["metrics_source"],
                "route": r.get("route"),
            }
            for r in records
        ],
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved {}", RESULTS_PATH)
    return report


def main() -> int:
    api = None
    for arg in sys.argv[1:]:
        if arg.startswith("--api="):
            api = arg.split("=", 1)[1]
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0
    try:
        report = run(api_base=api)
        summary = {
            "results": str(RESULTS_PATH),
            "tracks": report["tracks"],
            "overall_metrics": report["overall_metrics"],
        }
        print(json.dumps(summary, indent=2))
        abs_ok = report["tracks"]["abstention"].get("incorrect_refusals", 0) == 0
        ans_fail = report["tracks"]["answerable"].get("fail_count", 0)
        return 0 if abs_ok and ans_fail == 0 else 1
    except RagasScoringError as exc:
        logger.error("RAGAS failed: {}", exc)
        print(json.dumps({"error": "ragas_scoring_failed", "detail": str(exc)}, indent=2))
        return 2
    except Exception as exc:
        logger.exception("Evaluation failed: {}", exc)
        print(json.dumps({"error": "evaluation_failed", "detail": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
