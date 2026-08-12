"""FastAPI dependencies and small shared helpers.

V1 has no auth (ENVIRONMENT_SETUP.md defers auth to Phase 20 hardening), so
every investigation is attributed to one seeded system user/org/project.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from packages.domain.core import Organization, Project, User
from packages.domain.db import get_engine

SYSTEM_ORG_NAME = "Default Organization"
SYSTEM_PROJECT_NAME = "Default Project"
SYSTEM_USER_EMAIL = "engineer@local"


def get_db() -> Iterator[Session]:
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session


def get_or_create_default_project(session: Session) -> Project:
    org = session.query(Organization).filter_by(name=SYSTEM_ORG_NAME).one_or_none()
    if org is None:
        org = Organization(name=SYSTEM_ORG_NAME)
        session.add(org)
        session.flush()

    project = session.query(Project).filter_by(organization_id=org.id, name=SYSTEM_PROJECT_NAME).one_or_none()
    if project is None:
        project = Project(organization_id=org.id, name=SYSTEM_PROJECT_NAME, status="ACTIVE")
        session.add(project)
        session.flush()
    return project


def get_or_create_default_user(session: Session) -> User:
    org = session.query(Organization).filter_by(name=SYSTEM_ORG_NAME).one_or_none()
    if org is None:
        org = Organization(name=SYSTEM_ORG_NAME)
        session.add(org)
        session.flush()

    user = session.query(User).filter_by(email=SYSTEM_USER_EMAIL).one_or_none()
    if user is None:
        user = User(organization_id=org.id, email=SYSTEM_USER_EMAIL, display_name="Default Engineer", role="ENGINEER")
        session.add(user)
        session.flush()
    return user
