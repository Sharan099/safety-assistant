#!/usr/bin/env python3
"""Inspect SQLite migration state."""
import sqlite3
from pathlib import Path

db = Path("safety_registry.db")
if not db.exists():
    print("no db")
    raise SystemExit(1)

c = sqlite3.connect(db)
print("alembic", c.execute("SELECT version_num FROM alembic_version").fetchall())
print("chunk cols", [r[1] for r in c.execute("PRAGMA table_info(chunks)").fetchall()])
print("doc cols tail", [r[1] for r in c.execute("PRAGMA table_info(documents)").fetchall()][-6:])
print("ingest_log", c.execute("SELECT name FROM sqlite_master WHERE name='ingest_log'").fetchone())
