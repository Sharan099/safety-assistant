#!/usr/bin/env python3
"""Create the newly defined harness tables in SQLite without dropping existing data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger
from database.connection import engine
from database.models import Base


def main() -> None:
    logger.info("Running metadata creation for any missing tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Successfully created/verified harness tables!")


if __name__ == "__main__":
    main()
