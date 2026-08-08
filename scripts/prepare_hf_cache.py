#!/usr/bin/env python3
"""Stage Hugging Face model weights for offline Docker builds.

Run this on a machine that already has (or can reach) the model weights:

    python scripts/prepare_hf_cache.py

It populates ``docker/hf_cache/`` in standard HF Hub layout (``hub/models--*``)
so the Dockerfile can ``COPY`` them into the image with ``HF_HUB_OFFLINE=1``.

When a model is already in the local user cache (``~/.cache/huggingface``),
it is copied without re-downloading. Missing models are fetched once.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docker" / "hf_cache"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
MODELS = [EMBEDDING_MODEL, RERANKER_MODEL]


def _repo_cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _user_hub_dir() -> Path:
    return Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _dest_hub_dir(cache_dir: Path) -> Path:
    return cache_dir / "hub"


def _materialize_tree(src: Path, dest: Path) -> None:
    """Copy a directory tree, resolving symlinks (required on Windows)."""
    if not src.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dest / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.resolve(), target)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _copy_from_user_cache(repo_id: str, cache_dir: Path) -> bool:
    src = _user_hub_dir() / _repo_cache_name(repo_id)
    if not src.is_dir():
        return False
    dest = _dest_hub_dir(cache_dir) / _repo_cache_name(repo_id)
    print(f"  Materializing from {_user_hub_dir()}", flush=True)
    _materialize_tree(src, dest)
    return True


def _snapshot(repo_id: str, cache_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"\n=== {repo_id} -> {cache_dir} ===", flush=True)
    hub = _dest_hub_dir(cache_dir)
    hub.mkdir(parents=True, exist_ok=True)

    # Already staged from a prior prepare run?
    try:
        os.environ["HF_HOME"] = str(cache_dir)
        snapshot_download(repo_id=repo_id, local_files_only=True)
        print("  OK (already staged offline)", flush=True)
        return
    except Exception:
        pass

    if _copy_from_user_cache(repo_id, cache_dir):
        try:
            os.environ["HF_HOME"] = str(cache_dir)
            snapshot_download(repo_id=repo_id, local_files_only=True)
            print("  OK (reused local user cache)", flush=True)
            return
        except Exception as exc:
            print(f"  Local copy incomplete ({exc}) — will try Hub download", flush=True)

    print("  Downloading into user Hugging Face cache ...", flush=True)
    user_hf_home = Path.home() / ".cache" / "huggingface"
    os.environ["HF_HOME"] = str(user_hf_home)
    snapshot_download(repo_id=repo_id)
    if not _copy_from_user_cache(repo_id, cache_dir):
        raise FileNotFoundError(
            f"Downloaded {repo_id} but could not find it under {user_hf_home / 'hub'}"
        )
    os.environ["HF_HOME"] = str(cache_dir)
    snapshot_download(repo_id=repo_id, local_files_only=True)
    print("  OK (downloaded and materialized)", flush=True)


def _verify_offline(cache_dir: Path) -> None:
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("\n=== Offline verification (embedder) ===", flush=True)
    SentenceTransformer(EMBEDDING_MODEL)
    print("  Embedder OK", flush=True)

    print("\n=== Offline verification (reranker) ===", flush=True)
    AutoTokenizer.from_pretrained(RERANKER_MODEL)
    AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
    print("  Reranker OK", flush=True)


def main() -> int:
    # Staging must be allowed to reach the Hub when weights are missing.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    # Windows cannot create symlinks without Developer Mode / admin — copy files instead.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Staging HF cache at {DEST}", flush=True)

    errors: list[str] = []
    for model in MODELS:
        try:
            _snapshot(model, DEST)
        except Exception as exc:
            errors.append(f"{model}: {exc}")
            print(f"  FAILED: {exc}", flush=True)

    if errors:
        print(
            "\nERROR: Could not stage all models.\n" + "\n".join(f"  - {e}" for e in errors),
            file=sys.stderr,
            flush=True,
        )
        return 1

    try:
        _verify_offline(DEST)
    except Exception as exc:
        print(f"\nERROR: Offline verification failed: {exc}", file=sys.stderr, flush=True)
        return 1

    print(f"\nReady for docker build — {DEST} contains all required weights.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
