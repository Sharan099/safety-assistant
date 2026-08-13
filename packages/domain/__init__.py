"""Core domain model — see BACKEND_SCHEMA.md.

Import order matters for Alembic autogenerate: `base` first, then the entity
modules (each registers its tables on `Base.metadata` at import time).
"""

# Import every entity module so Base.metadata is fully populated for Alembic
# autogenerate and for anything that does `from packages.domain import Base`.
from packages.domain import copilot, core, investigation, knowledge, provenance  # noqa: F401,E402
from packages.domain.base import Base

__all__ = ["Base"]
