"""Source registry (allowlist) — `knowledge/00_registry/sources.yaml`.

Ingestion may only touch sources listed here, and every entry's local bytes
must match the recorded SHA-256 before anything reads them.
"""

from __future__ import annotations

import datetime
import pathlib
from functools import lru_cache
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Kind = Literal["REGULATION", "STANDARD", "TECHNICAL_REPORT", "MANUAL", "PROJECT_DOCUMENT"]

DEFAULT_REGISTRY_PATH = pathlib.Path("knowledge/00_registry/sources.yaml")


class VersionInfo(BaseModel):
    label: str
    series: str | None = None
    revision: str | None = None
    document_symbol: str | None = None
    published_at: datetime.date | None = None
    valid_from: datetime.date | None = None
    valid_to: datetime.date | None = None


class SourceEntry(BaseModel):
    source_key: str
    regulation_key: str
    kind: Kind
    title: str
    authority: str
    jurisdiction: str
    authority_level: str
    publisher: str | None = None
    data_class: str = "PUBLIC"
    source_uri: str | None = None
    source_uri_status: str = "UNKNOWN"
    local_path: str
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int
    media_type: str = "application/pdf"
    license: str = "UNVERIFIED"
    version: VersionInfo

    @field_validator("sha256")
    @classmethod
    def _lower_hex(cls, v: str) -> str:
        int(v, 16)  # raises if not hex
        return v.lower()


class SourceRegistry(BaseModel):
    schema_version: int
    sources: list[SourceEntry]

    def get(self, source_key: str) -> SourceEntry:
        for s in self.sources:
            if s.source_key == source_key:
                return s
        raise KeyError(f"source_key not in registry (allowlist): {source_key}")

    def keys(self) -> list[str]:
        return [s.source_key for s in self.sources]


def load_registry(path: pathlib.Path = DEFAULT_REGISTRY_PATH) -> SourceRegistry:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    reg = SourceRegistry.model_validate(raw)
    if reg.schema_version != 2:
        raise ValueError(f"unsupported registry schema_version {reg.schema_version}")
    keys = reg.keys()
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate source_key in registry")
    return reg


@lru_cache(maxsize=4)
def get_registry(path: str = str(DEFAULT_REGISTRY_PATH)) -> SourceRegistry:
    return load_registry(pathlib.Path(path))
