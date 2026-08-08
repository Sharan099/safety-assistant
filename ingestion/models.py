"""Shared ingestion data models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ContentType = Literal["clause", "table", "list", "text", "figure"]


class Chunk(BaseModel):
    """One structure-aware retrieval unit with grounding metadata.

    Payload always includes the Stage 0 metadata contract fields
    (``document_id``, ``section``, ``element_type``, ``parent_id``,
    ``element_id``, ``coordinates``) alongside legacy aliases used by
    retrieval (``content_type``, ``parent_section_id``, ``section_id``,
    ``bounding_box``, ``section_title``).
    """

    chunk_id: str
    text: str
    enriched_text: str = ""
    regulation_id: str
    revision: str
    section_number: str
    section_title: str
    page_number: int | None = None
    bounding_box: list[float] = Field(default_factory=list)
    content_type: ContentType = "clause"
    parent_section_id: str | None = None
    section_id: str
    heading_path: list[str] = Field(default_factory=list)
    ingested_at: str | None = None
    # Explicit Stage 0 contract fields (optional on construct; filled in payload).
    document_id: str | None = None
    element_id: str | None = None

    def payload(self) -> dict[str, Any]:
        """Qdrant payload: full metadata contract + legacy aliases + texts."""
        document_id = self.document_id or self.regulation_id
        element_id = self.element_id or self.section_id or self.chunk_id
        element_type = self.content_type
        parent_id = self.parent_section_id
        coordinates = list(self.bounding_box or [])
        data: dict[str, Any] = {
            # --- Stage 0 metadata contract ---
            "document_id": document_id,
            "page_number": self.page_number,
            "section": self.section_title,
            "section_number": self.section_number,
            "element_type": element_type,
            "parent_id": parent_id,
            "element_id": element_id,
            "coordinates": coordinates,
            "regulation_id": self.regulation_id,
            "revision": self.revision,
            # --- legacy / retrieval aliases ---
            "chunk_id": self.chunk_id,
            "text": self.text,
            "enriched_text": self.enriched_text or self.text,
            "section_title": self.section_title,
            "bounding_box": coordinates,
            "content_type": self.content_type,
            "parent_section_id": self.parent_section_id,
            "section_id": self.section_id,
            "heading_path": self.heading_path,
        }
        if self.ingested_at:
            data["ingested_at"] = self.ingested_at
        return data
