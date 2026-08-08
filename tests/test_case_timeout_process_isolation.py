"""Regression: per-case timeout kills only the worker subprocess.

Simulates a hung native call (slow reranker stand-in) via ``_test_hang``.
Confirms the harness logs ``timeout_or_native_crash`` and continues scoring
without crashing the parent process.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eval.case_timeout import CaseTimeoutPool, shutdown_case_timeout_pool


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_pool():
    shutdown_case_timeout_pool()
    yield
    shutdown_case_timeout_pool()


def test_hanging_case_kills_only_subprocess_and_harness_continues():
    """Hung worker is terminated; next case still runs in a fresh worker."""
    parent_pid = os.getpid()
    pool = CaseTimeoutPool(timeout_s=1.5)
    hung_case = {"id": "hang-1", "category": "factual_lookup", "question": "hang"}
    ok_case = {"id": "ok-1", "category": "factual_lookup", "question": "ok"}

    worker_before = pool._proc.pid if pool._proc is not None else None
    assert worker_before is not None

    # Stand-in for a native call that never returns (reranker / torch / onnx).
    failed = pool.run(
        {
            "case": hung_case,
            "_test_hang": True,
            "_test_hang_seconds": 3600,
        }
    )

    assert os.getpid() == parent_pid, "parent eval process must not die"
    assert failed.get("pass") is False
    assert failed.get("reason") == "timeout_or_native_crash"
    assert failed.get("id") == "hang-1"
    assert failed.get("timeout_or_native_crash") is True

    # Worker was recycled after the kill.
    assert pool._proc is not None and pool._proc.is_alive()
    worker_after = pool._proc.pid
    assert worker_after != worker_before

    # Harness continues: next case succeeds in the new worker.
    ok = pool.run({"case": ok_case, "_test_ok": True})
    assert os.getpid() == parent_pid
    assert ok.get("pass") is True
    assert ok.get("id") == "ok-1"
    assert ok.get("worker_pid") == worker_after

    pool.close()


def test_worker_soft_exception_does_not_kill_pool():
    pool = CaseTimeoutPool(timeout_s=10.0)
    pid0 = pool._proc.pid
    case = {"id": "boom-1", "category": "factual_lookup"}
    row = pool.run({"case": case, "_test_fn": "boom"})
    assert row.get("pass") is False
    assert row.get("reason") == "worker_exception"
    assert pool._proc is not None and pool._proc.is_alive()
    assert pool._proc.pid == pid0
    # Subsequent job still works on the same worker.
    ok = pool.run({"case": {"id": "ok-2"}, "_test_ok": True})
    assert ok.get("pass") is True
    pool.close()
