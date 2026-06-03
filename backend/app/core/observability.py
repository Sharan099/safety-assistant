"""LangSmith tracing for query, retrieval, prompt, response, and latency."""

import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

from loguru import logger

from backend.app.core.settings import (
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
)


def _configure_langsmith() -> None:
    if not LANGSMITH_TRACING or not LANGSMITH_API_KEY:
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)


_configure_langsmith()


def trace_run(name: str) -> Callable:
    """Decorator that logs to LangSmith when configured."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not LANGSMITH_TRACING or not LANGSMITH_API_KEY:
                return fn(*args, **kwargs)
            try:
                from langsmith import traceable

                return traceable(name=name)(fn)(*args, **kwargs)
            except Exception as exc:
                logger.warning(f"LangSmith trace skipped: {exc}")
                return fn(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def trace_span(
    name: str,
    inputs: dict | None = None,
) -> Generator[dict, None, None]:
    """Manual span for RAG pipeline steps."""
    run_data: dict = {"inputs": inputs or {}, "outputs": {}, "latency_ms": 0.0}
    t0 = time.perf_counter()
    try:
        yield run_data
    finally:
        run_data["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if LANGSMITH_TRACING and LANGSMITH_API_KEY:
            try:
                from langsmith import Client

                client = Client(api_key=LANGSMITH_API_KEY)
                client.create_run(
                    name=name,
                    run_type="chain",
                    inputs=run_data.get("inputs", {}),
                    outputs=run_data.get("outputs", {}),
                    project_name=LANGSMITH_PROJECT,
                    extra={"latency_ms": run_data["latency_ms"]},
                )
            except Exception as exc:
                logger.debug(f"LangSmith span failed: {exc}")
