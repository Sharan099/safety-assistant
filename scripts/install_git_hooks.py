#!/usr/bin/env python3
"""Install repo git hooks without requiring `git config core.hooksPath`.

Copies `.githooks/pre-commit` → `.git/hooks/pre-commit` so secret scanning
runs on every commit in this clone.
"""

from __future__ import annotations

import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".githooks" / "pre-commit"
DST = ROOT / ".git" / "hooks" / "pre-commit"


def main() -> int:
    if not SRC.is_file():
        print(f"missing hook source: {SRC}", file=sys.stderr)
        return 1
    if not (ROOT / ".git").exists():
        print("not a git checkout (.git missing)", file=sys.stderr)
        return 1
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    mode = DST.stat().st_mode
    DST.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed {DST.relative_to(ROOT)}")
    print("secret pattern check will run on git commit (staged files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
