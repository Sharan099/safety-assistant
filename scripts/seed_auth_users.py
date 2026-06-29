#!/usr/bin/env python3
"""Create auth tables and seed named users for session login (Phase A0)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger
from database.connection import SessionLocal, engine
from database.models import Base, User
from registry.auth import hash_password

DEFAULT_USERS = (
    ("engineer_a", "engineer_a"),
    ("engineer_b", "engineer_b"),
    ("lead", "lead"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed session-auth users")
    parser.add_argument(
        "--password",
        default=os.getenv("AUTH_SEED_PASSWORD", "changeme"),
        help="Password for all seeded users (override via AUTH_SEED_PASSWORD)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Replace password hashes for existing seeded usernames",
    )
    args = parser.parse_args()

    if args.password == "changeme":
        logger.warning("Using default password 'changeme' — set AUTH_SEED_PASSWORD in production")

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    created = 0
    updated = 0
    try:
        for user_id, username in DEFAULT_USERS:
            existing = db.query(User).filter(User.user_id == user_id).first()
            pwd_hash = hash_password(args.password)
            if existing:
                if args.reset:
                    existing.password_hash = pwd_hash
                    updated += 1
                    logger.info(f"Updated password for {username}")
                else:
                    logger.info(f"User {username} already exists (skip)")
                continue
            db.add(User(user_id=user_id, username=username, password_hash=pwd_hash))
            created += 1
            logger.info(f"Created user {username} (user_id={user_id})")
        db.commit()
    finally:
        db.close()

    print(f"Done: {created} created, {updated} updated. Login with username + AUTH_SEED_PASSWORD.")


if __name__ == "__main__":
    main()
