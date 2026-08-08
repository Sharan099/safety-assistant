"""Per-case wall-clock timeout via a persistent *process* worker.

Why process, not thread
-----------------------
Forcibly interrupting a thread that is inside numpy/torch/onnx (local reranker,
embeddings, RAGAS internals) is unsafe on Windows and is a known cause of
``0xC0000005`` access violations. A timed-out child process can be
``terminate()``/``kill()``'d without corrupting the parent eval harness.

Design
------
A single long-lived worker process (spawn context) loads models once on first
use, pulls jobs from a queue, and returns results on another queue. On timeout
or native crash the worker is killed and a fresh one is started — the harness
records ``reason="timeout_or_native_crash"`` and continues.

Tradeoff
--------
- Safe kill isolation (required on Windows with native libs).
- Models are reloaded only when the worker is recycled (timeout/crash), not
  per case — persistent pool avoids per-case spawn+reload cost on the happy path.
- Still pays one process-spawn when the pool is first created / recycled.

Disable with ``EVAL_CASE_TIMEOUT=0`` (synchronous in-process scoring).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from queue import Empty
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_CASE_TIMEOUT_S = 900.0  # 15 minutes — covers slow RAGAS + SUT
_SENTINEL = ("__stop__", None)


def case_timeout_enabled() -> bool:
    raw = (os.getenv("EVAL_CASE_TIMEOUT") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def case_timeout_seconds() -> float:
    try:
        return float(os.getenv("EVAL_CASE_TIMEOUT_S") or DEFAULT_CASE_TIMEOUT_S)
    except ValueError:
        return DEFAULT_CASE_TIMEOUT_S


def timeout_failure_row(
    case: dict[str, Any],
    *,
    reason: str = "timeout_or_native_crash",
    detail: str = "",
) -> dict[str, Any]:
    """Canonical failed-case payload when the worker is killed or dies."""
    cat = str(case.get("category") or "").strip().lower() or None
    return {
        "id": case.get("id"),
        "category": cat,
        "severity": case.get("severity"),
        "question": case.get("question"),
        "pass": False,
        "error": reason,
        "reason": reason,
        "detail": detail or reason,
        "timeout_or_native_crash": True,
    }


def _score_payload_in_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one case inside the child process (models load lazily here)."""
    # Test hooks (no RAG / native libs) — used by regression tests only.
    if payload.get("_test_ok"):
        return {
            "id": (payload.get("case") or {}).get("id"),
            "pass": True,
            "ok": True,
            "worker_pid": os.getpid(),
        }

    if payload.get("_test_hang"):
        # Stand-in for a hung native call (reranker / torch / onnx).
        hang_s = float(payload.get("_test_hang_seconds") or 3600)
        time.sleep(hang_s)
        return {"id": (payload.get("case") or {}).get("id"), "pass": True, "hung": True}

    if payload.get("_test_fn") == "boom":
        raise RuntimeError("simulated_worker_exception")

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    from eval.case_scoring import score_one_case
    from generation.llm_client import LLMClient

    case = payload["case"]
    llm = LLMClient(
        provider=payload.get("llm_provider"),
        use_cache=bool(payload.get("llm_use_cache", False)),
    )
    # Rebuild optional security judges/guards inside the worker when requested.
    security_judge = None
    guard_input = None
    guard_output = None
    if payload.get("with_security"):
        try:
            from eval.scoring.security_scorer import (
                build_input_guards,
                build_output_guards,
                build_security_judge,
            )

            security_judge = build_security_judge()
            guard_input = build_input_guards()
            guard_output = build_output_guards()
        except Exception as exc:  # noqa: BLE001
            logger.warning("security judge/guards unavailable in worker: %s", exc)

    return score_one_case(
        case,
        llm=llm,
        skip_ragas=bool(payload.get("skip_ragas", False)),
        security_judge=security_judge,
        guard_input=guard_input,
        guard_output=guard_output,
    )


def _persistent_worker(job_q: Any, result_q: Any) -> None:
    """Long-lived child: pull (job_id, payload) → put (job_id, result|error)."""
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    while True:
        item = job_q.get()
        if not item or item[0] == _SENTINEL[0]:
            break
        job_id, payload = item
        try:
            result = _score_payload_in_worker(payload)
            result_q.put((job_id, {"ok": True, "result": result}))
        except BaseException as exc:  # noqa: BLE001 — report then keep serving
            result_q.put(
                (
                    job_id,
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
            )


@dataclass
class CaseTimeoutPool:
    """Persistent single-worker process pool with hard kill on deadline."""

    timeout_s: float = DEFAULT_CASE_TIMEOUT_S
    _ctx: Any = None
    _job_q: Any = None
    _result_q: Any = None
    _proc: Any = None

    def __post_init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._start_worker()

    def _start_worker(self) -> None:
        self._job_q = self._ctx.Queue()
        self._result_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_persistent_worker,
            args=(self._job_q, self._result_q),
            name="eval-case-worker",
            daemon=True,
        )
        self._proc.start()
        logger.info(
            "case timeout worker started pid=%s timeout_s=%.0f",
            self._proc.pid,
            self.timeout_s,
        )

    def _kill_worker(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.is_alive():
            logger.warning("terminating timed-out/crashed case worker pid=%s", proc.pid)
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            proc.join(timeout=5)
            if proc.is_alive():
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                proc.join(timeout=5)
        self._proc = None

    def recycle(self) -> None:
        """Kill the current worker and start a fresh one (models reload once)."""
        self._kill_worker()
        # Drain stale queues so a late result cannot leak into the next job.
        self._job_q = None
        self._result_q = None
        self._start_worker()

    def close(self) -> None:
        try:
            if self._job_q is not None and self._proc is not None and self._proc.is_alive():
                self._job_q.put(_SENTINEL)
                self._proc.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self._kill_worker()

    def run(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Run one payload in the worker; on timeout kill worker and return failure row."""
        timeout = float(timeout_s if timeout_s is not None else self.timeout_s)
        case = payload.get("case") or {}
        if self._proc is None or not self._proc.is_alive():
            self.recycle()

        job_id = uuid.uuid4().hex
        assert self._job_q is not None and self._result_q is not None
        self._job_q.put((job_id, payload))
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                rid, envelope = self._result_q.get(timeout=min(0.5, remaining))
            except Empty:
                if self._proc is not None and not self._proc.is_alive():
                    exitcode = getattr(self._proc, "exitcode", None)
                    self.recycle()
                    return timeout_failure_row(
                        case,
                        reason="timeout_or_native_crash",
                        detail=f"worker_exited exitcode={exitcode}",
                    )
                continue
            if rid != job_id:
                # Stale result from a previous recycled worker — ignore.
                continue
            if envelope.get("ok"):
                return envelope["result"]
            # Soft exception inside worker — process still healthy; do not recycle.
            err = str(envelope.get("error") or "worker_exception")
            cat = str(case.get("category") or "").strip().lower() or None
            return {
                "id": case.get("id"),
                "category": cat,
                "severity": case.get("severity"),
                "question": case.get("question"),
                "pass": False,
                "error": err,
                "reason": "worker_exception",
                "detail": err,
            }

        # Deadline exceeded — kill only the worker subprocess.
        self.recycle()
        return timeout_failure_row(
            case,
            reason="timeout_or_native_crash",
            detail=f"exceeded_case_timeout_s={timeout}",
        )


_POOL: CaseTimeoutPool | None = None


def get_case_timeout_pool(*, timeout_s: float | None = None) -> CaseTimeoutPool:
    global _POOL
    seconds = float(timeout_s if timeout_s is not None else case_timeout_seconds())
    if _POOL is None:
        _POOL = CaseTimeoutPool(timeout_s=seconds)
    elif abs(_POOL.timeout_s - seconds) > 1e-6:
        _POOL.timeout_s = seconds
    return _POOL


def shutdown_case_timeout_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


def score_case_with_timeout(
    case: dict[str, Any],
    *,
    skip_ragas: bool = False,
    with_security: bool = True,
    llm_provider: str | None = None,
    timeout_s: float | None = None,
    sync_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score one eval case under a hard wall-clock deadline.

    When ``EVAL_CASE_TIMEOUT=0``, runs ``sync_fn`` (or in-process scoring) with
    no process isolation — useful for debugging.
    """
    if not case_timeout_enabled():
        if sync_fn is not None:
            return sync_fn()
        return _score_payload_in_worker(
            {
                "case": case,
                "skip_ragas": skip_ragas,
                "with_security": with_security,
                "llm_provider": llm_provider,
            }
        )

    pool = get_case_timeout_pool(timeout_s=timeout_s)
    return pool.run(
        {
            "case": case,
            "skip_ragas": skip_ragas,
            "with_security": with_security,
            "llm_provider": llm_provider,
        },
        timeout_s=timeout_s,
    )
