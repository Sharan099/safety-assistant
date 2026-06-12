"""
Lightweight persistence layer (SQLite, stdlib only).

Stores users, chat sessions, messages, and feedback for the testing phase so we
can analyse failures and improve the system. Designed to be safe for a small
multi-user cloud deployment:

  - All queries are parameterized (no string interpolation -> no SQL injection).
  - A process-wide lock serialises writes (SQLite is single-writer).
  - check_same_thread=False so the FastAPI threadpool can share one connection.

NOTE: SQLite is fine for a low-traffic testing phase. For higher concurrency or
multi-instance cloud deployment, point APP_DB_PATH at a shared volume or migrate
to Postgres (the SQL here is intentionally portable).
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from config import OUTPUT_DIR

_DB_PATH = Path(os.getenv("APP_DB_PATH", str(OUTPUT_DIR / "app.db")))
_USERNAME_RE = re.compile(r"^[A-Za-z0-9 _.\-]{2,40}$")

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def _now() -> int:
    return int(time.time())


def _new_id() -> str:
    return uuid.uuid4().hex


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                _conn = sqlite3.connect(
                    str(_DB_PATH), check_same_thread=False, timeout=30
                )
                _conn.row_factory = sqlite3.Row
                _conn.execute("PRAGMA journal_mode=WAL;")
                _init_schema(_conn)
                logger.info(f"Feedback store ready at {_DB_PATH}")
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            username    TEXT UNIQUE NOT NULL,
            created_at  INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            created_at  INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id            TEXT PRIMARY KEY,
            session_id    TEXT,
            user_id       TEXT,
            query         TEXT,
            answer        TEXT,
            grounding     TEXT,
            model         TEXT,
            created_at    INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id            TEXT PRIMARY KEY,
            message_id    TEXT,
            session_id    TEXT,
            user_id       TEXT,
            rating        TEXT NOT NULL,
            reasons       TEXT,
            comment       TEXT,
            query         TEXT,
            answer        TEXT,
            created_at    INTEGER NOT NULL
        );
        """
    )
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent, backward-safe schema migrations for existing databases."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
    if "model" not in cols:
        # Pre-v3.0 databases lack the model column used for routing analytics.
        conn.execute("ALTER TABLE messages ADD COLUMN model TEXT")
        logger.info("store: migrated messages table (+model column)")


def is_valid_username(username: str) -> bool:
    return bool(_USERNAME_RE.match((username or "").strip()))


def get_or_create_user(username: str) -> dict[str, Any]:
    """Idempotent: returns the existing user or creates a new one."""
    username = (username or "").strip()
    if not is_valid_username(username):
        raise ValueError(
            "Username must be 2-40 chars (letters, numbers, space, _ . -)."
        )
    conn = _connect()
    with _lock:
        row = conn.execute(
            "SELECT id, username, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row:
            return {**dict(row), "created": False}
        uid = _new_id()
        ts = _now()
        conn.execute(
            "INSERT INTO users (id, username, created_at) VALUES (?, ?, ?)",
            (uid, username, ts),
        )
        conn.commit()
        return {"id": uid, "username": username, "created_at": ts, "created": True}


def create_session(user_id: str, session_id: str | None = None) -> str:
    conn = _connect()
    sid = session_id or _new_id()
    with _lock:
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO sessions (id, user_id, created_at) VALUES (?, ?, ?)",
                (sid, user_id or "", _now()),
            )
            conn.commit()
    return sid


def record_message(
    *,
    session_id: str | None,
    user_id: str | None,
    query: str,
    answer: str,
    grounding: str | None = None,
    model: str | None = None,
) -> str:
    """Persist a Q/A turn and return its message id (for feedback linkage)."""
    conn = _connect()
    mid = _new_id()
    with _lock:
        conn.execute(
            "INSERT INTO messages "
            "(id, session_id, user_id, query, answer, grounding, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (mid, session_id, user_id, query, answer, grounding, model, _now()),
        )
        conn.commit()
    return mid


def session_depth(session_id: str | None) -> int:
    """Number of prior turns in a session (a gateway routing input)."""
    if not session_id:
        return 0
    conn = _connect()
    with _lock:
        row = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
    return int(row[0]) if row else 0


def recent_downvote_rate(user_id: str | None, *, limit: int = 20) -> float:
    """Share of the user's recent feedback that was 👎 (0..1). Routing input #8."""
    if not user_id:
        return 0.0
    conn = _connect()
    with _lock:
        rows = conn.execute(
            "SELECT rating FROM feedback WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    if not rows:
        return 0.0
    down = sum(1 for r in rows if r["rating"] == "down")
    return round(down / len(rows), 4)


def model_performance() -> dict[str, float]:
    """Per-model success proxy (1 - downvote_rate) from feedback joined to the
    answering model. Routing input #9 (previous model performance)."""
    conn = _connect()
    with _lock:
        rows = conn.execute(
            """
            SELECT m.model AS model,
                   SUM(CASE WHEN f.rating='down' THEN 1 ELSE 0 END) AS downs,
                   COUNT(f.id) AS total
            FROM feedback f
            JOIN messages m ON m.id = f.message_id
            WHERE m.model IS NOT NULL
            GROUP BY m.model
            """
        ).fetchall()
    perf: dict[str, float] = {}
    for r in rows:
        total = int(r["total"] or 0)
        if total <= 0:
            continue
        downs = int(r["downs"] or 0)
        perf[r["model"]] = round(1.0 - downs / total, 4)
    return perf


def record_feedback(
    *,
    message_id: str | None,
    session_id: str | None,
    user_id: str | None,
    rating: str,
    reasons: list[str] | None,
    comment: str | None,
    query: str | None,
    answer: str | None,
) -> str:
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    conn = _connect()
    fid = _new_id()
    reasons_str = "|".join(r.strip() for r in (reasons or []) if r.strip())[:1000]
    with _lock:
        conn.execute(
            "INSERT INTO feedback "
            "(id, message_id, session_id, user_id, rating, reasons, comment, "
            " query, answer, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fid,
                message_id,
                session_id,
                user_id,
                rating,
                reasons_str,
                (comment or "")[:4000],
                (query or "")[:4000],
                (answer or "")[:8000],
                _now(),
            ),
        )
        conn.commit()
    return fid


def feedback_stats() -> dict[str, Any]:
    conn = _connect()
    with _lock:
        up = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating='up'"
        ).fetchone()[0]
        down = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating='down'"
        ).fetchone()[0]
        users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    return {
        "users": users,
        "messages": msgs,
        "thumbs_up": up,
        "thumbs_down": down,
    }
