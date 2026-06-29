#!/usr/bin/env python3
"""Assemble a minimal HF Space deploy tree and push to sharan099/Passive_safety_assistant."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "_hf_deploy"
HF_REMOTE = "https://huggingface.co/spaces/sharan099/Passive_safety_assistant"

COPY_FILES = [
    "Dockerfile",
    ".dockerignore",
    ".gitattributes",
    "alembic.ini",
    "coverage_expected.yaml",
    "requirements.txt",
]
COPY_DIRS = [
    "app",
    "api",
    "database",
    "parser",
    "registry",
    "vectorization",
    "scheduler",
    "backend",
    "regulation_discovery",
    "monitoring",
    "docker",
    "scripts",
]
DATA_HF = ["README.md"]  # DB fetched from GitHub LFS at Docker build time

README = """---
title: Passive Safety Registry API
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Passive Safety Registry API

Session auth, confidential upload, hybrid retrieval.
GitHub: https://github.com/Sharan099/safety-assistant
"""


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def _ignore_pycache(_dir: str, names: list[str]) -> list[str]:
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]


def main() -> None:
    if DEPLOY.exists():
        shutil.rmtree(DEPLOY)
    DEPLOY.mkdir()

    for name in COPY_FILES:
        shutil.copy2(ROOT / name, DEPLOY / name)

    for d in COPY_DIRS:
        if d == "backend":
            continue
        shutil.copytree(ROOT / d, DEPLOY / d, ignore=_ignore_pycache)

    shutil.copytree(ROOT / "backend", DEPLOY / "backend", ignore=_ignore_pycache)

    data_hf = DEPLOY / "data" / "hf"
    data_hf.mkdir(parents=True)
    for name in DATA_HF:
        src = ROOT / "data" / "hf" / name
        if src.is_file():
            shutil.copy2(src, data_hf / name)

    (DEPLOY / "README.md").write_text(README, encoding="utf-8")
    print("Deploy bundle ready (code only — DB pulled from GitHub at build)", flush=True)

    run(["git", "init"], DEPLOY)
    run(["git", "lfs", "install"], DEPLOY)
    run(["git", "add", "."], DEPLOY)
    run(["git", "commit", "-m", "Registry app HF Space deploy"], DEPLOY)
    run(["git", "branch", "-M", "main"], DEPLOY)
    run(["git", "remote", "add", "hf", HF_REMOTE], DEPLOY)
    run(["git", "push", "hf", "main:main", "--force"], DEPLOY)
    print("Pushed to HF Space successfully.", flush=True)


if __name__ == "__main__":
    main()
