"""ARCHIVED STUB — scoring removed.

The orphan HTTP/API RAGAS stack lived here (formerly ``python -m evaluation.run``).
``evaluation/scoring.py`` was deleted once CI moved to ``eval.run_full`` +
``eval.scoring.ragas_scorer`` / ``security_scorer``. This file remains only as a
one-cycle breadcrumb so a mistaken ``python -m evaluation.run`` fails loudly.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print(
        "evaluation.run was archived and its scoring module removed.\n"
        "Use: python -m eval.run_full\n"
        "See archive/evaluation/README.md",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
