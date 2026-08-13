"""Copilot conversation persistence — PRD_COPILOT_UPDATE.md §9,
CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 3.

One conversation per investigation (no global memory in V1.1, per both
source docs) — simpler than a conversation-selection UI/API, and matches
"every conversation must belong to investigation_id."
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from packages.domain.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class CopilotConversation(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "copilot_conversations"

    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id"), unique=True, index=True
    )


class CopilotMessage(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "copilot_messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("copilot_conversations.id"), index=True
    )

    # user | assistant | system
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)

    # Evidence.id values this message cites — PRD_COPILOT_UPDATE.md §9.
    evidence_refs: Mapped[list[str] | None] = mapped_column(JSONB)
    # PRD_COPILOT_UPDATE.md §10: suggested_actions / unknowns on the response.
    suggested_actions: Mapped[list[str] | None] = mapped_column(JSONB)
    unknowns: Mapped[list[str] | None] = mapped_column(JSONB)
    # Set when the LLM was unavailable and this message fell back to a
    # deterministic/templated response — TRD.md §30, PRD_COPILOT_UPDATE.md §11.
    llm_degraded: Mapped[bool] = mapped_column(default=False)


class CopilotToolCall(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "copilot_tool_calls"

    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("copilot_messages.id"), index=True)

    tool_name: Mapped[str] = mapped_column(Text)
    arguments: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # SUCCESS | FAILURE
    status: Mapped[str] = mapped_column(Text)
    result_summary: Mapped[str | None] = mapped_column(Text)

    started_at: Mapped[datetime.datetime] = mapped_column()
    completed_at: Mapped[datetime.datetime | None] = mapped_column()
