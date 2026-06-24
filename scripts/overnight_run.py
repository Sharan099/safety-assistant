#!/usr/bin/env python3
"""
Overnight pipeline: verify chunks -> embed -> evaluate -> compare results.
Stops at the first failed stage. Logs to output/overnight_run.log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    EVALUATION_ARCHIVE_V31,
    EVALUATION_CURRENT,
    OVERNIGHT_LOG,
)

LOG_FILE = OVERNIGHT_LOG
OLD_RESULTS = EVALUATION_ARCHIVE_V31 / "rag_eval_20_results.json"
NEW_RESULTS = EVALUATION_CURRENT / "rag_eval_20_results.json"
RAGAS_KEYS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
CONDA_ENV = os.getenv("OVERNIGHT_CONDA_ENV", "rag")


def _rag_python() -> str:
    """Resolve rag-env Python. Prefer an active `conda activate rag` session."""
    if os.getenv("CONDA_DEFAULT_ENV") == CONDA_ENV:
        return sys.executable

    override = os.getenv("RAG_PYTHON", "").strip()
    if override:
        return override

    conda_exe = os.environ.get("CONDA_EXE", "").strip()
    bases: list[Path] = []
    if conda_exe:
        bases.append(Path(conda_exe).resolve().parent.parent)
    for candidate in (Path.home() / "anaconda3", Path.home() / "miniconda3"):
        if candidate.is_dir():
            bases.append(candidate)

    py_name = "python.exe" if os.name == "nt" else "bin/python"
    for base in bases:
        py = base / "envs" / CONDA_ENV / py_name
        if py.is_file():
            return str(py)

    raise RuntimeError(
        f"Could not find conda env '{CONDA_ENV}'. "
        f"Run `conda activate {CONDA_ENV}` then retry."
    )


def _python_cmd(script: str) -> list[str]:
    """Run Python in the rag conda env (never `conda run` — segfaults on Windows)."""
    return [_rag_python(), script]


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _stream_line(line: str, logf) -> None:
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write(line.encode(enc, errors="replace"))
        sys.stdout.buffer.flush()
    logf.write(line)


def run_cmd(cmd: list[str], stage: str, extra_env: dict | None = None) -> int:
    log(f"STAGE {stage} — running: {' '.join(cmd)}")
    env = {**os.environ, **(extra_env or {})}
    with open(LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write(f"[{ts()}] --- subprocess output begin ({stage}) ---\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _stream_line(line, logf)
        logf.flush()
        return proc.wait()


def stage1_verify_chunks() -> bool:
    log("STAGE 1 START — verify chunk source")
    if not CHUNKS_FILE.exists():
        log(f"STAGE 1 FAILED — {CHUNKS_FILE} does not exist")
        return False

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks", [])
    total = data.get("total_chunks") or len(chunks)
    with_text = [c for c in chunks if (c.get("text") or "").strip()]
    unique_ids = len({c["chunk_id"] for c in with_text})
    dup_rows = len(with_text) - unique_ids

    log(f"STAGE 1 — total_chunks={total}, unique_chunk_ids={unique_ids}")
    if dup_rows:
        log(f"STAGE 1 FAILED — {dup_rows} duplicate chunk_id row(s); re-run hierarchical_chunker")
        return False

    if data.get("unique_chunk_ids") and data["unique_chunk_ids"] != unique_ids:
        log(
            f"STAGE 1 FAILED — metadata unique_chunk_ids={data['unique_chunk_ids']} "
            f"!= actual {unique_ids}"
        )
        return False

    log("STAGE 1 END — OK")
    return True


def stage2_embed() -> bool:
    log("STAGE 2 START — resumable embedding")
    embed_env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "EMBEDDING_BATCH": os.getenv("EMBEDDING_BATCH", "2"),
        "EMBED_SAVE_EVERY": os.getenv("EMBED_SAVE_EVERY", "200"),
    }
    log(
        f"STAGE 2 — env: EMBEDDING_BATCH={embed_env['EMBEDDING_BATCH']}, "
        f"EMBED_SAVE_EVERY={embed_env['EMBED_SAVE_EVERY']}, conda env={CONDA_ENV}"
    )
    rc = run_cmd(
        _python_cmd(str(ROOT / "ingestion" / "embed_chunks.py")),
        "2-embed",
        extra_env=embed_env,
    )
    if rc != 0:
        log(f"STAGE 2 FAILED — embed_chunks.py exited with code {rc}")
        return False

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks_with_text = [
        c for c in chunks_data.get("chunks", []) if (c.get("text") or "").strip()
    ]
    expected_rows = len(chunks_with_text)
    expected_unique = len({c["chunk_id"] for c in chunks_with_text})

    if not EMBEDDINGS_FILE.exists():
        log(f"STAGE 2 FAILED — {EMBEDDINGS_FILE} not found after embed")
        return False

    with open(EMBEDDINGS_FILE, encoding="utf-8") as f:
        emb_data = json.load(f)
    total_vectors = emb_data.get("total_vectors", len(emb_data.get("embeddings", {})))
    log(
        f"STAGE 2 — total_vectors={total_vectors}, "
        f"expected_unique_ids={expected_unique} ({expected_rows} chunk rows)"
    )

    if total_vectors != expected_unique:
        log(
            f"STAGE 2 FAILED — vector count mismatch: "
            f"total_vectors={total_vectors}, unique_chunk_ids={expected_unique}"
        )
        return False

    log("STAGE 2 END — OK")
    return True


def stage3_evaluate() -> bool:
    log("STAGE 3 START — 20-question evaluation (live LLM, no proxy fallback on rate limit)")
    env = dict(**{k: v for k, v in os.environ.items()})
    env["EVAL_RESULTS_NAME"] = NEW_RESULTS.name
    env.pop("EVAL_SKIP_LLM", None)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    log(f"STAGE 3 — EVAL_RESULTS_NAME={env['EVAL_RESULTS_NAME']}, EVAL_SKIP_LLM unset")

    with open(LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write(f"[{ts()}] --- subprocess output begin (3-eval) ---\n")
        logf.flush()
        proc = subprocess.Popen(
            _python_cmd(str(ROOT / "tests" / "run_full_evaluation.py")),
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _stream_line(line, logf)
        logf.write(f"[{ts()}] --- subprocess output end (3-eval) ---\n")
        logf.flush()
        rc = proc.wait()

    if rc != 0:
        log(f"STAGE 3 FAILED — run_full_evaluation.py exited with code {rc}")
        return False

    if not NEW_RESULTS.exists():
        log(f"STAGE 3 FAILED — results file not written: {NEW_RESULTS}")
        return False

    log("STAGE 3 END — OK")
    return True


def stage4_compare() -> bool:
    log("STAGE 4 START — save label and comparison table")

    if not NEW_RESULTS.exists():
        log(f"STAGE 4 FAILED — {NEW_RESULTS} missing")
        return False

    with open(NEW_RESULTS, encoding="utf-8") as f:
        new_data = json.load(f)
    new_ragas = new_data.get("ragas") or new_data.get("ragas_proxies") or {}

    old_ragas: dict = {}
    if OLD_RESULTS.exists():
        with open(OLD_RESULTS, encoding="utf-8") as f:
            old_data = json.load(f)
        old_ragas = old_data.get("ragas") or old_data.get("ragas_proxies") or {}
    else:
        log(f"WARNING — old results not found at {OLD_RESULTS}; deltas will be N/A")

    header = f"{'Metric':<22} {'Old (v3.1)':>12} {'New (v3.2)':>12} {'Δ':>10}"
    sep = "-" * len(header)
    lines = [
        "",
        "RAGAS comparison — embedding + text_section chunking upgrade (20Q)",
        sep,
        header,
        sep,
    ]
    for key in RAGAS_KEYS:
        old_v = old_ragas.get(key)
        new_v = new_ragas.get(key)
        label = key.replace("_", " ").title()
        if old_v is not None and new_v is not None:
            delta = float(new_v) - float(old_v)
            lines.append(
                f"{label:<22} {float(old_v):>12.3f} {float(new_v):>12.3f} {delta:>+10.3f}"
            )
        else:
            lines.append(f"{label:<22} {'N/A':>12} {new_v if new_v is not None else 'N/A':>12} {'N/A':>10}")

    old_overall = (
        sum(float(old_ragas[k]) for k in RAGAS_KEYS if old_ragas.get(k) is not None)
        / max(sum(1 for k in RAGAS_KEYS if old_ragas.get(k) is not None), 1)
    )
    new_overall = new_data.get("overall_score")
    if new_overall is None and new_ragas:
        new_overall = sum(float(new_ragas[k]) for k in RAGAS_KEYS if new_ragas.get(k) is not None) / len(RAGAS_KEYS)
    if new_overall is not None and old_ragas:
        delta_o = float(new_overall) - old_overall
        lines.append(sep)
        lines.append(
            f"{'Overall (mean)':<22} {old_overall:>12.3f} {float(new_overall):>12.3f} {delta_o:>+10.3f}"
        )

    lines.append(sep)
    lines.append(f"New results: {NEW_RESULTS}")
    lines.append(f"Old results: {OLD_RESULTS}")
    lines.append(f"Eval mode:   {new_data.get('evaluation_mode', 'unknown')}")

    for line in lines:
        log(line)

    log("STAGE 4 END — OK")
    return True


def main() -> int:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = LOG_FILE.parent / "overnight_run.lock"
    if lock.exists():
        try:
            old_pid = int(lock.read_text(encoding="utf-8").strip())
            log(f"WARNING: lock file exists (pid {old_pid}); waiting 5s before proceeding")
            import time
            time.sleep(5)
        except ValueError:
            pass
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return _main_impl()
    finally:
        lock.unlink(missing_ok=True)


def _main_impl() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("OVERNIGHT PIPELINE START")
    log(f"Log file: {LOG_FILE}")
    log("=" * 60)

    if not stage1_verify_chunks():
        log("PIPELINE HALTED at STAGE 1")
        return 1

    if not stage2_embed():
        log("PIPELINE HALTED at STAGE 2")
        return 1

    if not stage3_evaluate():
        log("PIPELINE HALTED at STAGE 3")
        return 1

    if not stage4_compare():
        log("PIPELINE HALTED at STAGE 4")
        return 1

    log("=" * 60)
    log("OVERNIGHT PIPELINE COMPLETE — all stages succeeded")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
