"""Conversation session state: last N (question, answer) turns + active vehicle context."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "conversations.sqlite3"
DEFAULT_MAX_TURNS = 6

_lock = threading.Lock()
_db_path: Path | None = None


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


@dataclass
class VehicleContext:
    """Active vehicle/platform context carried across turns (BMW multi-turn workflow)."""

    categories: list[str] = field(default_factory=list)
    powertrain: str | None = None  # ev | hybrid | ice
    mass_kg: float | None = None
    model_hints: list[str] = field(default_factory=list)
    seating_hint: str | None = None
    platform: str | None = None

    def has_any(self) -> bool:
        return bool(
            self.categories
            or self.powertrain
            or self.mass_kg is not None
            or self.model_hints
            or self.seating_hint
            or self.platform
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")}

    def format_suffix(self) -> str:
        """Short parenthetical for condensation / retrieval (not a full restate)."""
        bits: list[str] = []
        if self.platform:
            bits.append(self.platform)
        elif self.model_hints:
            bits.append(self.model_hints[0])
        if self.categories:
            bits.append("/".join(self.categories))
        if self.powertrain:
            bits.append(self.powertrain.upper() if self.powertrain != "ice" else "ICE")
        if self.mass_kg is not None:
            bits.append(f"{self.mass_kg:g} kg")
        if self.seating_hint:
            bits.append(self.seating_hint)
        if not bits:
            return ""
        return f"(vehicle: {', '.join(bits)})"


def _path() -> Path:
    global _db_path
    if _db_path is None:
        _db_path = Path(os.getenv("CONVERSATIONS_DB") or DEFAULT_DB)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    return _db_path


def _max_turns() -> int:
    try:
        return max(1, int(os.getenv("CONVERSATION_MAX_TURNS", str(DEFAULT_MAX_TURNS))))
    except ValueError:
        return DEFAULT_MAX_TURNS


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    conversation_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (conversation_id, turn_index)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_meta (
                    conversation_id TEXT PRIMARY KEY,
                    vehicle_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def get_turns(conversation_id: str, *, limit: int | None = None) -> list[Turn]:
    """Return up to ``limit`` most recent turns (oldest → newest)."""
    cid = (conversation_id or "").strip()
    if not cid:
        return []
    init_db()
    n = limit if limit is not None else _max_turns()
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT question, answer FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY turn_index DESC
                LIMIT ?
                """,
                (cid, n),
            ).fetchall()
        finally:
            conn.close()
    turns = [Turn(question=r["question"], answer=r["answer"]) for r in reversed(rows)]
    return turns


def append_turn(conversation_id: str, question: str, answer: str) -> None:
    """Append a turn and trim to the last N."""
    cid = (conversation_id or "").strip()
    if not cid:
        return
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q:
        return
    init_db()
    max_n = _max_turns()
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) AS m FROM conversation_turns WHERE conversation_id = ?",
                (cid,),
            ).fetchone()
            nxt = int(row["m"]) + 1
            conn.execute(
                """
                INSERT INTO conversation_turns
                (conversation_id, turn_index, question, answer, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cid, nxt, q, a, time.time()),
            )
            # Drop older turns beyond window.
            conn.execute(
                """
                DELETE FROM conversation_turns
                WHERE conversation_id = ?
                  AND turn_index <= ?
                """,
                (cid, nxt - max_n),
            )
            conn.commit()
        finally:
            conn.close()


def clear_conversation(conversation_id: str) -> None:
    cid = (conversation_id or "").strip()
    if not cid:
        return
    init_db()
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "DELETE FROM conversation_turns WHERE conversation_id = ?", (cid,)
            )
            conn.execute(
                "DELETE FROM conversation_meta WHERE conversation_id = ?", (cid,)
            )
            conn.commit()
        finally:
            conn.close()


def get_vehicle_context(conversation_id: str) -> VehicleContext:
    cid = (conversation_id or "").strip()
    if not cid:
        return VehicleContext()
    init_db()
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT vehicle_json FROM conversation_meta WHERE conversation_id = ?",
                (cid,),
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return VehicleContext()
    try:
        data = json.loads(row["vehicle_json"] or "{}")
    except json.JSONDecodeError:
        return VehicleContext()
    if not isinstance(data, dict):
        return VehicleContext()
    return VehicleContext(
        categories=[str(x) for x in (data.get("categories") or []) if str(x).strip()],
        powertrain=(str(data["powertrain"]).strip() if data.get("powertrain") else None),
        mass_kg=float(data["mass_kg"]) if data.get("mass_kg") is not None else None,
        model_hints=[str(x) for x in (data.get("model_hints") or []) if str(x).strip()],
        seating_hint=(
            str(data["seating_hint"]).strip() if data.get("seating_hint") else None
        ),
        platform=(str(data["platform"]).strip() if data.get("platform") else None),
    )


def set_vehicle_context(conversation_id: str, ctx: VehicleContext) -> None:
    cid = (conversation_id or "").strip()
    if not cid or not ctx.has_any():
        return
    init_db()
    payload = json.dumps(ctx.to_public_dict(), ensure_ascii=False)
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO conversation_meta (conversation_id, vehicle_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    vehicle_json = excluded.vehicle_json,
                    updated_at = excluded.updated_at
                """,
                (cid, payload, time.time()),
            )
            conn.commit()
        finally:
            conn.close()


def merge_vehicle_context(
    existing: VehicleContext,
    incoming: VehicleContext,
) -> VehicleContext:
    """Non-null / non-empty fields from ``incoming`` overlay ``existing``."""
    return VehicleContext(
        categories=list(incoming.categories) or list(existing.categories),
        powertrain=incoming.powertrain or existing.powertrain,
        mass_kg=incoming.mass_kg if incoming.mass_kg is not None else existing.mass_kg,
        model_hints=list(incoming.model_hints) or list(existing.model_hints),
        seating_hint=incoming.seating_hint or existing.seating_hint,
        platform=incoming.platform
        or existing.platform
        or (incoming.model_hints[0] if incoming.model_hints else None)
        or (existing.model_hints[0] if existing.model_hints else None),
    )


def update_vehicle_context_from_question(
    conversation_id: str,
    question: str,
) -> VehicleContext:
    """Parse vehicle cues from the question, merge with stored context, persist."""
    from retrieval.applicability import parse_vehicle_profile

    cid = (conversation_id or "").strip()
    existing = get_vehicle_context(cid) if cid else VehicleContext()
    profile = parse_vehicle_profile(question)
    incoming = VehicleContext(
        categories=list(profile.categories),
        powertrain=profile.powertrain,
        mass_kg=profile.mass_kg,
        model_hints=list(profile.model_hints),
        seating_hint=profile.seating_hint,
        platform=profile.model_hints[0] if profile.model_hints else None,
    )
    # Also scan prior turns when the new question is a short follow-up.
    if cid and not incoming.has_any():
        for turn in get_turns(cid):
            prior = parse_vehicle_profile(turn.question)
            if prior.categories or prior.powertrain or prior.model_hints:
                incoming = VehicleContext(
                    categories=list(prior.categories),
                    powertrain=prior.powertrain,
                    mass_kg=prior.mass_kg,
                    model_hints=list(prior.model_hints),
                    seating_hint=prior.seating_hint,
                    platform=prior.model_hints[0] if prior.model_hints else None,
                )
                break
    merged = merge_vehicle_context(existing, incoming)
    if cid and merged.has_any():
        set_vehicle_context(cid, merged)
    return merged


def inject_vehicle_context(question: str, ctx: VehicleContext | None) -> str:
    """Append active vehicle suffix when the question omits it."""
    q = (question or "").strip()
    if not q or ctx is None or not ctx.has_any():
        return q
    suffix = ctx.format_suffix()
    if not suffix:
        return q
    # Already present?
    low = q.lower()
    markers = [
        *(c.lower() for c in ctx.categories),
        (ctx.powertrain or "").lower(),
        *((m.lower() for m in ctx.model_hints)),
        (ctx.platform or "").lower(),
        "vehicle:",
    ]
    if any(m and m in low for m in markers):
        return q
    return f"{q.rstrip(' ?')} {suffix}?"


def format_history_for_prompt(turns: Sequence[Turn]) -> str:
    if not turns:
        return "(no prior turns)"
    lines: list[str] = []
    for i, t in enumerate(turns, 1):
        lines.append(f"Turn {i} user: {t.question}")
        ans = (t.answer or "").strip()
        if len(ans) > 400:
            ans = ans[:400] + "…"
        lines.append(f"Turn {i} assistant: {ans}")
    return "\n".join(lines)
