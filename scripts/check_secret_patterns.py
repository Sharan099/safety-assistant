#!/usr/bin/env python3
"""Block commits that embed live API keys or credential-like strings.

Used as a pre-commit hook (see .githooks/pre-commit). Scans staged files by
default, or paths passed on the CLI.

  python scripts/check_secret_patterns.py              # staged files
  python scripts/check_secret_patterns.py path [path…] # explicit paths
  python scripts/check_secret_patterns.py --self-test
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# High-signal provider token shapes (length floors reduce false positives).
TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Groq API key (gsk_…)", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("OpenRouter key (sk-or-v1-…)", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{20,}\b")),
    ("OpenAI-style key (sk-…)", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("Google API key (AIza…)", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b")),
    ("Hugging Face token (hf_…)", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("NVIDIA API key (nvapi-…)", re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}\b")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    (
        "DB URL with embedded password",
        re.compile(
            r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://"
            r"[^\s:/@]+:[^\s@/]{4,}@"
        ),
    ),
]

# set/export/assign of known secret env vars to a non-placeholder value
_SECRET_ENV = (
    r"(?:GROQ_API_KEY|GOOGLE_API_KEY|NVIDIA_API_KEY|OPENROUTER_API_KEY|"
    r"HF_TOKEN|HUGGING_FACE_HUB_TOKEN|QDRANT_API_KEY|DATABASE_URL|"
    r"POSTGRES_PASSWORD|DB_PASSWORD|DATABASE_PASSWORD)"
)
_ASSIGN = re.compile(
    rf"(?i)(?:set\s+[\"']?|export\s+|)\s*{_SECRET_ENV}\s*[=:]\s*[\"']?"
    rf"([^\s\"'&;]+)"
)

_PLACEHOLDER = re.compile(
    r"(?i)^(?:"
    r"<.*>|"
    r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|"
    r"%[A-Za-z_][A-Za-z0-9_]*%|"
    r"changeme|replace.?me|your[-_].+|xxx+|todo|none|null|example|"
    r"dummy|placeholder|redacted|not.?set|insert.+"
    r"|\.\.\.|…"
    r")$"
)

# Skip binary / huge / vendor trees that are never meant to hold launcher secrets
_SKIP_DIR_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    "docker/hf_cache",
}
_SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".pyc",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".bin",
    ".pt",
    ".onnx",
    ".woff",
    ".woff2",
}


def _is_skipped(path: Path) -> bool:
    parts = set(path.as_posix().split("/"))
    if parts & _SKIP_DIR_PARTS:
        return True
    if path.suffix.lower() in _SKIP_SUFFIXES:
        return True
    return False


def _staged_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    files: list[Path] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        p = ROOT / line
        if p.is_file() and not _is_skipped(Path(line)):
            files.append(p)
    return files


def _looks_like_placeholder(value: str) -> bool:
    v = value.strip().strip("\"'")
    if len(v) < 8:
        return True
    if _PLACEHOLDER.fullmatch(v):
        return True
    # Common local/dev non-secrets
    if v.startswith(("http://", "https://", "localhost", "./", "../")):
        return True
    return False


def scan_text(text: str, source: str) -> list[str]:
    hits: list[str] = []
    for label, pat in TOKEN_PATTERNS:
        for m in pat.finditer(text):
            # Never echo the secret — only location + rule name
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append(f"{source}:{line_no}: matched {label}")
    for m in _ASSIGN.finditer(text):
        value = m.group(1)
        if _looks_like_placeholder(value):
            continue
        line_no = text.count("\n", 0, m.start()) + 1
        hits.append(f"{source}:{line_no}: matched secret env assignment with non-placeholder value")
    return hits


def scan_file(path: Path) -> list[str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"{path}: could not read ({exc})"]
    if b"\0" in raw[:8192]:
        return []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    rel = path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()
    return scan_text(text, rel)


def self_test() -> int:
    cases: list[tuple[str, bool]] = [
        ("export GROQ_API_KEY=gsk_" + ("a" * 40), True),
        ("OPENROUTER_API_KEY=sk-or-v1-" + ("b" * 40), True),
        ("GOOGLE_API_KEY=AIza" + ("c" * 35), True),
        ("HF_TOKEN=hf_" + ("d" * 30), True),
        ("NVIDIA_API_KEY=nvapi-" + ("e" * 40), True),
        ("postgres://user:s3cretpass@localhost:5432/db", True),
        ('set "GROQ_API_KEY=gsk_' + ("f" * 40) + '"', True),
        ("GROQ_API_KEY=", False),
        ("GROQ_API_KEY=${GROQ_API_KEY}", False),
        ("GROQ_API_KEY=changeme", False),
        ("# GROQ_API_KEY=", False),
        ("PORTKEY_GATEWAY_URL=http://localhost:8787/v1", False),
    ]
    failed = 0
    for text, expect_hit in cases:
        hits = scan_text(text, "self-test")
        got = bool(hits)
        if got != expect_hit:
            print(f"FAIL expect_hit={expect_hit} got={got}: {text[:48]}…", file=sys.stderr)
            failed += 1
    if failed:
        print(f"self-test: {failed} failure(s)", file=sys.stderr)
        return 1
    print("self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Files to scan (default: git staged)")
    parser.add_argument("--self-test", action="store_true", help="Run built-in pattern checks")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.paths:
        files = [Path(p) for p in args.paths if Path(p).is_file() and not _is_skipped(Path(p))]
    else:
        files = _staged_files()

    if not files:
        return 0

    all_hits: list[str] = []
    for f in files:
        all_hits.extend(scan_file(f if f.is_absolute() else ROOT / f))

    if not all_hits:
        return 0

    print("ERROR: commit blocked — possible secrets detected:\n", file=sys.stderr)
    for h in all_hits:
        print(f"  {h}", file=sys.stderr)
    print(
        "\nRemove the credentials (use .env / CI secrets) and try again.\n"
        "If this is a false positive, narrow the string or ask a maintainer.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
