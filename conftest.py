"""Pytest bootstrap: ensure the repo root is importable.

This lets tests use absolute imports (`from backend.app... import ...` and
`from config import ...`) exactly as the application code does.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
