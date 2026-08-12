"""Core entities — BACKEND_SCHEMA.md §2, §6-17.

Organization, User, Project, Vehicle, ModelVersion, Component,
ComponentRevision, SimulationRun, Artifact, SignalDefinition, Signal,
SignalFeature.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import Double, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from packages.domain.base import Base, CreatedAtMixin, UpdatedAtMixin, UUIDPrimaryKeyMixin


class Organization(Base, UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin):
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(Text)


class User(Base, UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin):
    __tablename__ = "users"

    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"), index=True)
    email: Mapped[str] = mapped_column(Text, unique=True)
    display_name: Mapped[str] = mapped_column(Text)
    # ENGINEER | REVIEWER | ADMIN
    role: Mapped[str] = mapped_column(Text)


class Project(Base, UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin):
    __tablename__ = "projects"

    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"), index=True)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="ACTIVE")


class Vehicle(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "vehicles"

    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(Text)
    programme: Mapped[str | None] = mapped_column(Text)
    vehicle_type: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class ModelVersion(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "model_versions"

    vehicle_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("vehicles.id"), index=True)
    version: Mapped[str] = mapped_column(Text)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("model_versions.id"))
    source_artifact_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("artifacts.id"))
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


# COMPONENT_TYPES: BODY | SEAT | BELT | AIRBAG | DUMMY | MATERIAL | CONTACT |
#                  CONNECTOR | STRUCTURE | SOLVER_CONTROL | OTHER
class Component(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "components"

    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(Text)
    component_type: Mapped[str] = mapped_column(Text)
    parent_component_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("components.id"))
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class ComponentRevision(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "component_revisions"

    component_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("components.id"), index=True)
    revision: Mapped[str] = mapped_column(Text)
    source_artifact_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("artifacts.id"))
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class SimulationRun(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "simulation_runs"

    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    vehicle_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("vehicles.id"), index=True)
    model_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_versions.id"), index=True)

    run_id: Mapped[str] = mapped_column(Text, unique=True)
    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("simulation_runs.id"))

    solver: Mapped[str | None] = mapped_column(Text)
    solver_version: Mapped[str | None] = mapped_column(Text)

    dummy_version: Mapped[str | None] = mapped_column(Text)
    impact_type: Mapped[str | None] = mapped_column(Text)
    impact_speed: Mapped[float | None] = mapped_column(Double)
    barrier: Mapped[str | None] = mapped_column(Text)

    seat_configuration: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    restraint_configuration: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    result_processing_version: Mapped[str | None] = mapped_column(Text)

    status: Mapped[str] = mapped_column(Text, default="UNKNOWN")
    # PASS | WARNING | FAIL | UNKNOWN — PRD.md PR-003
    quality_status: Mapped[str] = mapped_column(Text, default="UNKNOWN")

    started_at: Mapped[datetime.datetime | None] = mapped_column()
    completed_at: Mapped[datetime.datetime | None] = mapped_column()

    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class Artifact(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "artifacts"

    # use_alter: artifacts -> simulation_runs -> model_versions -> artifacts
    # (via ModelVersion.source_artifact_id) is a genuine cycle in the schema
    # (BACKEND_SCHEMA.md §13/§14). Both FKs are nullable, so defer this one
    # to an ALTER TABLE after every table exists, instead of reworking the
    # documented schema. See docs/ADR/0005.
    simulation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulation_runs.id", use_alter=True, name="fk_artifacts_simulation_run_id"),
        index=True,
    )

    artifact_type: Mapped[str] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(Text)
    storage_uri: Mapped[str] = mapped_column(Text)

    mime_type: Mapped[str | None] = mapped_column(Text)
    size_bytes: Mapped[int | None] = mapped_column()
    sha256: Mapped[str] = mapped_column(Text, unique=True)

    source_type: Mapped[str | None] = mapped_column(Text)


class SignalDefinition(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "signal_definitions"
    __table_args__ = (UniqueConstraint("canonical_name", name="uq_signal_definitions_canonical_name"),)

    name: Mapped[str] = mapped_column(Text)
    canonical_name: Mapped[str] = mapped_column(Text)
    domain: Mapped[str | None] = mapped_column(Text)
    unit: Mapped[str | None] = mapped_column(Text)
    quantity: Mapped[str | None] = mapped_column(Text)
    source_standard: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class Signal(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "signals"

    simulation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("simulation_runs.id"), index=True
    )
    signal_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signal_definitions.id"), index=True
    )

    name: Mapped[str] = mapped_column(Text)
    source_channel: Mapped[str | None] = mapped_column(Text)
    storage_uri: Mapped[str] = mapped_column(Text)  # Parquet/HDF5 — raw samples never live in Postgres

    sampling_rate: Mapped[float | None] = mapped_column(Double)
    start_time: Mapped[float | None] = mapped_column(Double)
    end_time: Mapped[float | None] = mapped_column(Double)
    unit: Mapped[str | None] = mapped_column(Text)

    processing_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class SignalFeature(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "signal_features"

    signal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("signals.id"), index=True)

    feature_name: Mapped[str] = mapped_column(Text)
    value: Mapped[float] = mapped_column(Double)
    unit: Mapped[str | None] = mapped_column(Text)

    algorithm_version: Mapped[str] = mapped_column(Text)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
