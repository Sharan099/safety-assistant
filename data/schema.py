"""
schema.py — UN R14 Occupant Protection Knowledge Graph Schema

Refactored from aviation ATA GraphRAG schema into:
- Passive safety
- Occupant protection
- Homologation engineering
- UN R14 regulatory knowledge graph

Primary regulation:
UN Regulation No. 14 — Safety Belt Anchorages

Key engineering domains:
- Seat belt anchorage systems
- Vehicle homologation
- Crashworthiness validation
- Restraint systems
- Geometry constraints
- Static and dynamic testing

Author:
Sharan — Occupant Protection GraphRAG

"""

import re
from enum import Enum
from typing import Optional, Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class EntityType(str, Enum):

    # Regulation hierarchy
    REGULATION = "Regulation"
    REGULATION_SECTION = "RegulationSection"
    ANNEX = "Annex"

    # Vehicle safety domain
    VEHICLE_CATEGORY = "VehicleCategory"
    SEAT = "Seat"
    SEAT_TYPE = "SeatType"

    BELT_ANCHORAGE = "BeltAnchorage"
    EFFECTIVE_BELT_ANCHORAGE = "EffectiveBeltAnchorage"

    SAFETY_BELT = "SafetyBelt"

    # Geometry and constraints
    GEOMETRY_CONSTRAINT = "GeometryConstraint"
    ANGLE_REQUIREMENT = "AngleRequirement"
    DISTANCE_REQUIREMENT = "DistanceRequirement"
    POSITION_REQUIREMENT = "PositionRequirement"

    # Testing
    TEST_PROCEDURE = "TestProcedure"
    STATIC_TEST = "StaticTest"
    DYNAMIC_TEST = "DynamicTest"

    TEST_LOAD = "TestLoad"
    TEST_CONFIGURATION = "TestConfiguration"

    # Compliance
    REQUIREMENT = "Requirement"
    COMPLIANCE_RULE = "ComplianceRule"
    APPROVAL_REQUIREMENT = "ApprovalRequirement"

    # Measurements
    MEASUREMENT = "Measurement"
    LOAD_CASE = "LoadCase"
    FORCE = "Force"
    ANGLE = "Angle"

    # Validation
    VALIDATION_RESULT = "ValidationResult"
    FAILURE_MODE = "FailureMode"

    # Materials
    MATERIAL_SPECIFICATION = "MaterialSpecification"

    # Graph infrastructure
    CHUNK = "Chunk"
    COMMUNITY = "Community"


class RelationshipType(str, Enum):

    # Structural
    CONTAINS = "CONTAINS"
    PART_OF = "PART_OF"

    # Regulation
    DEFINES = "DEFINES"
    REQUIRES = "REQUIRES"
    GOVERNS = "GOVERNS"
    APPLIES_TO = "APPLIES_TO"

    # Testing
    TESTS = "TESTS"
    VALIDATES = "VALIDATES"
    LOADS = "LOADS"
    MEASURES = "MEASURES"

    # Geometry
    CONSTRAINS = "CONSTRAINS"
    LOCATED_AT = "LOCATED_AT"
    POSITIONED_RELATIVE_TO = "POSITIONED_RELATIVE_TO"

    # Compliance
    SATISFIES = "SATISFIES"
    VIOLATES = "VIOLATES"
    CERTIFIES = "CERTIFIES"

    # Safety
    PROTECTS = "PROTECTS"
    RESTRAINS = "RESTRAINS"
    TRANSFERS_LOAD_TO = "TRANSFERS_LOAD_TO"

    # Graph
    SOURCED_FROM = "SOURCED_FROM"
    MEMBER_OF = "MEMBER_OF"


class TestType(str, Enum):
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"


class SeatOrientation(str, Enum):
    FORWARD = "FORWARD"
    REARWARD = "REARWARD"
    SIDE = "SIDE"


class VehicleClass(str, Enum):
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"


# ═══════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    """
    Normalize text for deterministic IDs.
    """

    replacements = {
        "ä": "AE",
        "ö": "OE",
        "ü": "UE",
        "ß": "SS"
    }

    for char, repl in replacements.items():
        name = name.replace(char, repl)

    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"[\s\-_]+", "_", name.strip())

    return name.upper()


def normalize_for_dedup(name: str) -> str:
    n = normalize_name(name).lower()
    return re.sub(r"_+", "", n)


# ═══════════════════════════════════════════════════════════════════════
# ID GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def make_requirement_id(section: str, name: str) -> str:
    return f"REQ-{section}-{normalize_name(name)}"


def make_test_id(section: str, test_name: str) -> str:
    return f"TEST-{section}-{normalize_name(test_name)}"


def make_load_id(section: str, load: float) -> str:
    return f"LOAD-{section}-{int(load)}"


def make_anchor_id(location: str, anchor_type: str) -> str:
    return f"ANCH-{normalize_name(location)}-{normalize_name(anchor_type)}"


def make_geometry_id(parameter: str) -> str:
    return f"GEO-{normalize_name(parameter)}"


def make_seat_id(name: str) -> str:
    return f"SEAT-{normalize_name(name)}"


# ═══════════════════════════════════════════════════════════════════════
# PROVENANCE
# ═══════════════════════════════════════════════════════════════════════

class Provenance(BaseModel):

    regulation: str = "UN_R14"

    section: Optional[str] = None

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    source_chunk: Optional[str] = None

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0
    )


# ═══════════════════════════════════════════════════════════════════════
# BASE ENTITY
# ═══════════════════════════════════════════════════════════════════════

class BaseEntity(BaseModel):

    id: str

    type: EntityType

    provenance: Provenance = Field(
        default_factory=Provenance
    )

    model_config = {
        "populate_by_name": True,
        "extra": "ignore"
    }

    def dedup_key(self):
        return self.id


# ═══════════════════════════════════════════════════════════════════════
# REGULATION ENTITIES
# ═══════════════════════════════════════════════════════════════════════

class Regulation(BaseEntity):

    type: EntityType = EntityType.REGULATION

    regulation_number: str

    title: str

    revision: Optional[str] = None


class RegulationSection(BaseEntity):

    type: EntityType = EntityType.REGULATION_SECTION

    section_number: str

    title: str

    description: Optional[str] = None


class Annex(BaseEntity):

    type: EntityType = EntityType.ANNEX

    annex_number: str

    title: str


# ═══════════════════════════════════════════════════════════════════════
# VEHICLE DOMAIN ENTITIES
# ═══════════════════════════════════════════════════════════════════════

class VehicleCategory(BaseEntity):

    type: EntityType = EntityType.VEHICLE_CATEGORY

    category: VehicleClass

    description: Optional[str] = None


class Seat(BaseEntity):

    type: EntityType = EntityType.SEAT

    seat_name: str

    orientation: SeatOrientation

    adjustable: bool = False

    has_headrest: bool = False

    seat_position: Optional[str] = None

    vehicle_category: Optional[str] = None

    def dedup_key(self):
        return f"seat::{normalize_for_dedup(self.seat_name)}"


class BeltAnchorage(BaseEntity):

    type: EntityType = EntityType.BELT_ANCHORAGE

    anchorage_name: str

    anchorage_type: str

    seat_reference: Optional[str] = None

    threaded_size: Optional[str] = None

    structural_location: Optional[str] = None

    x_position: Optional[float] = None
    y_position: Optional[float] = None
    z_position: Optional[float] = None

    def dedup_key(self):
        return f"anchorage::{normalize_for_dedup(self.anchorage_name)}"


class SafetyBelt(BaseEntity):

    type: EntityType = EntityType.SAFETY_BELT

    belt_type: str

    has_retractor: bool = False

    has_pretensioner: bool = False

    load_limiter: bool = False


# ═══════════════════════════════════════════════════════════════════════
# GEOMETRY CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

class GeometryConstraint(BaseEntity):

    type: EntityType = EntityType.GEOMETRY_CONSTRAINT

    parameter_name: str

    min_value: Optional[float] = None
    max_value: Optional[float] = None

    unit: str

    applicable_vehicle: Optional[str] = None

    regulation_reference: Optional[str] = None

    description: Optional[str] = None

    def dedup_key(self):
        return f"geometry::{normalize_for_dedup(self.parameter_name)}"


class AngleRequirement(BaseEntity):

    type: EntityType = EntityType.ANGLE_REQUIREMENT

    angle_name: str

    min_angle: float

    max_angle: float

    unit: str = "degree"

    applicable_seat: Optional[str] = None


class DistanceRequirement(BaseEntity):

    type: EntityType = EntityType.DISTANCE_REQUIREMENT

    parameter_name: str

    min_distance: Optional[float] = None

    max_distance: Optional[float] = None

    unit: str = "mm"

    regulation_reference: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# TEST ENTITIES
# ═══════════════════════════════════════════════════════════════════════

class TestProcedure(BaseEntity):

    type: EntityType = EntityType.TEST_PROCEDURE

    test_name: str

    test_type: TestType

    regulation_reference: Optional[str] = None

    vehicle_category: Optional[str] = None

    application_angle: Optional[float] = None

    hold_time_seconds: Optional[float] = None

    description: Optional[str] = None


class StaticTest(BaseEntity):

    type: EntityType = EntityType.STATIC_TEST

    test_name: str

    load_application_time: Optional[float] = None

    hold_time_seconds: Optional[float] = None


class DynamicTest(BaseEntity):

    type: EntityType = EntityType.DYNAMIC_TEST

    test_name: str

    crash_speed: Optional[float] = None

    dummy_type: Optional[str] = None


class TestLoad(BaseEntity):

    type: EntityType = EntityType.TEST_LOAD

    load_value: float

    tolerance: Optional[float] = None

    unit: str = "daN"

    test_configuration: Optional[str] = None

    regulation_reference: Optional[str] = None

    def dedup_key(self):
        return f"load::{self.load_value}{self.unit}"


class TestConfiguration(BaseEntity):

    type: EntityType = EntityType.TEST_CONFIGURATION

    configuration_name: str

    description: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════

class Requirement(BaseEntity):

    type: EntityType = EntityType.REQUIREMENT

    requirement_text: str

    regulation_reference: str

    pass_criteria: Optional[str] = None

    failure_criteria: Optional[str] = None

    applicable_vehicle: Optional[str] = None

    def dedup_key(self):
        return f"requirement::{normalize_for_dedup(self.requirement_text[:80])}"


class ComplianceRule(BaseEntity):

    type: EntityType = EntityType.COMPLIANCE_RULE

    rule_text: str

    regulation_section: str

    pass_condition: Optional[str] = None

    failure_condition: Optional[str] = None


class ApprovalRequirement(BaseEntity):

    type: EntityType = EntityType.APPROVAL_REQUIREMENT

    approval_text: str

    approval_authority: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# MEASUREMENTS
# ═══════════════════════════════════════════════════════════════════════

class Measurement(BaseEntity):

    type: EntityType = EntityType.MEASUREMENT

    parameter_name: str

    value: float

    unit: str

    tolerance: Optional[float] = None

    condition: Optional[str] = None

    def dedup_key(self):
        return (
            f"measurement::"
            f"{normalize_for_dedup(self.parameter_name)}"
            f"-{self.value}-{self.unit}"
        )


class Force(BaseEntity):

    type: EntityType = EntityType.FORCE

    value: float

    unit: str = "daN"


class Angle(BaseEntity):

    type: EntityType = EntityType.ANGLE

    value: float

    unit: str = "degree"


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════

class ValidationResult(BaseEntity):

    type: EntityType = EntityType.VALIDATION_RESULT

    test_reference: str

    passed: bool

    measured_value: Optional[float] = None

    limit_value: Optional[float] = None

    comments: Optional[str] = None


class FailureMode(BaseEntity):

    type: EntityType = EntityType.FAILURE_MODE

    failure_name: str

    description: Optional[str] = None

    severity: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# RELATIONSHIPS
# ═══════════════════════════════════════════════════════════════════════

class Relationship(BaseModel):

    source: str

    target: str

    type: RelationshipType

    properties: dict[str, Any] = Field(default_factory=dict)

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0
    )

    source_chunk: Optional[str] = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v):
        return str(v).upper().replace(" ", "_")


# ═══════════════════════════════════════════════════════════════════════
# EXTRACTION OUTPUT
# ═══════════════════════════════════════════════════════════════════════

class ExtractionResult(BaseModel):

    chunk_id: str

    entities: list[dict] = Field(default_factory=list)

    relationships: list[Relationship] = Field(default_factory=list)

    extraction_ms: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════
# PROMPT SCHEMA
# ═══════════════════════════════════════════════════════════════════════

def build_extraction_schema_prompt() -> str:

    return '''
Return JSON in this structure:

{
  "entities": [
    {
      "id": "REQ-5_4_2-ALPHA1",
      "type": "GeometryConstraint",
      "parameter_name": "alpha_1",
      "min_value": 30,
      "max_value": 80,
      "unit": "degree"
    }
  ],

  "relationships": [
    {
      "source": "TEST-6_4_1-THREE_POINT_STATIC",
      "target": "LOAD-6_4_1-1350",
      "type": "REQUIRES"
    }
  ]
}
'''


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    req = GeometryConstraint(
        id=make_geometry_id("alpha_1"),
        parameter_name="alpha_1",
        min_value=30,
        max_value=80,
        unit="degree",
        applicable_vehicle="M1",
        regulation_reference="5.4.2.1"
    )

    test = TestProcedure(
        id=make_test_id("6_4_1", "three_point_static"),
        test_name="Three Point Belt Static Test",
        test_type=TestType.STATIC,
        regulation_reference="6.4.1"
    )

    load = TestLoad(
        id=make_load_id("6_4_1", 1350),
        load_value=1350,
        tolerance=20,
        unit="daN",
        regulation_reference="6.4.1.2"
    )

    rel = Relationship(
        source=test.id,
        target=load.id,
        type=RelationshipType.REQUIRES
    )

    print(req.model_dump())
    print(test.model_dump())
    print(load.model_dump())
    print(rel.model_dump())