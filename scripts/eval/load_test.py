"""Closed-loop load test against a running API — p50/p95/p99, throughput, errors.

    uv run uvicorn safety_assistant.api.main:app --port 8010 &
    uv run python scripts/eval/load_test.py --base http://localhost:8010 --users 5 --seconds 60

Queries are drawn from the gold dataset so the mix is realistic. Results are
written to evals/results/load_<timestamp>.json with git SHA and settings.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import random
import statistics
import subprocess
import sys
import threading
import time

import httpx

from safety_assistant.evaluation import load_dataset


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8010")
    ap.add_argument("--endpoint", default="/api/v1/search", choices=["/api/v1/search", "/api/v1/ask"])
    ap.add_argument("--users", type=int, default=5)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--token", default="")
    ap.add_argument("--dataset", default="evals/datasets/regulatory_v1.yaml")
    ap.add_argument("--out", default="evals/results")
    args = ap.parse_args(argv)

    queries = [c.query for c in load_dataset(pathlib.Path(args.dataset)).cases if c.answerability == "answerable"]
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    latencies: list[float] = []
    errors: dict[str, int] = {}
    throttled = [0]
    lock = threading.Lock()
    deadline = time.perf_counter() + args.seconds

    def worker(seed: int) -> None:
        rng = random.Random(seed)
        with httpx.Client(base_url=args.base, timeout=60.0, headers=headers) as client:
            while time.perf_counter() < deadline:
                q = rng.choice(queries)
                t0 = time.perf_counter()
                try:
                    r = client.post(args.endpoint, json={"query": q, "k": 8})
                    ok = r.status_code == 200
                    key = str(r.status_code)
                    if r.status_code == 429:  # throttled: back off, count separately
                        time.sleep(float(r.headers.get("retry-after", "1")))
                        with lock:
                            throttled[0] += 1
                        continue
                except httpx.HTTPError as exc:
                    ok, key = False, type(exc).__name__
                dt = time.perf_counter() - t0
                with lock:
                    if ok:
                        latencies.append(dt)
                    else:
                        errors[key] = errors.get(key, 0) + 1

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.users)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_start

    n = len(latencies)
    report = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None,
        "endpoint": args.endpoint,
        "users": args.users,
        "seconds": round(elapsed, 1),
        "requests_ok": n,
        "errors": errors,
        "throttled_429": throttled[0],
        "rps": round(n / elapsed, 2) if elapsed else 0,
        "latency_ms": {
            "p50": round(_pct(latencies, 0.50) * 1000, 1),
            "p95": round(_pct(latencies, 0.95) * 1000, 1),
            "p99": round(_pct(latencies, 0.99) * 1000, 1),
            "mean": round(statistics.fmean(latencies) * 1000, 1) if latencies else 0,
            "max": round(max(latencies) * 1000, 1) if latencies else 0,
        },
        "error_rate": round(sum(errors.values()) / max(1, n + sum(errors.values())), 4),
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"load_{report['timestamp'].replace(':', '').replace('-', '')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"written: {path}")
    return 0 if report["error_rate"] < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
