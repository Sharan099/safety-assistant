"""Configure PyTorch / OpenMP thread pools before ML model load."""

from __future__ import annotations

import os

import torch
from loguru import logger


def _default_thread_count() -> int:
    return max(1, os.cpu_count() or 4)


def configure_torch_threads() -> int:
    """Set OMP/MKL env vars and torch thread pools (call once at process startup)."""
    threads = int(os.getenv("OMP_NUM_THREADS", str(_default_thread_count())))
    threads = max(1, threads)

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))

    torch.set_num_threads(threads)
    interop = min(threads, 4)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass

    logger.info(
        "Torch threads configured: omp={} torch={} interop={}",
        threads,
        torch.get_num_threads(),
        interop,
    )
    return threads
