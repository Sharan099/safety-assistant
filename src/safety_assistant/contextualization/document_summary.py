"""One generated retrieval summary per document version.

Cache key = sha256(artifact sha256 + prompt version + model): an unchanged PDF is never
re-summarised on restart or re-ingest; a changed PDF, prompt or model produces exactly one
new summary. A failed generation is recorded (status FAILED) so retrieval falls back to the
identity block + original chunk and a later run can retry. Provider data-class policy is
applied here too: a document the configured LLM is not cleared to see is SKIPPED, not sent.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from safety_assistant.contextualization.context_builder import document_context
from safety_assistant.contextualization.prompts import (
    SUMMARY_MAX_TOKENS,
    SUMMARY_PROMPT_VERSION,
    SUMMARY_SYSTEM,
    summary_user_message,
)
from safety_assistant.persistence.models import DocumentSummary, Regulation, RegulationVersion, Section
from safety_assistant.providers.llm import LLMError, LLMMessage, LLMProvider, LLMResponse

log = logging.getLogger(__name__)

EXCERPT_FRONT_CHARS = 2500
EXCERPT_OUTLINE_LINES = 60
EXCERPT_BODY_CHARS = 5000
MIN_SUMMARY_CHARS = 80
MAX_SUMMARY_CHARS = 2500
# Bounded retry for retryable provider errors (429 / 5xx / timeout) — free-tier gateways rate-limit
# bursts; a summary is one call per document, so waiting is cheaper than a FAILED row.
RETRY_DELAYS_S = (5.0, 15.0, 30.0)
# Free-tier models sometimes return their reasoning ("The user wants me to…") or markdown scaffolding
# instead of the summary. Those are rejected deterministically and recorded as FAILED (retryable):
# a bad summary would be indexed into every chunk of the document.
_NOT_A_SUMMARY = re.compile(
    r"^(the user|we need|i need|i should|i will|i'll|let me|let's|okay|ok,|sure|here is|here's|first,|as an ai"
    r"|thinking process|analysis:|reasoning:)",
    re.IGNORECASE,
)


def validate_summary(text: str, finish_reason: str | None) -> str | None:
    """Reason the text is unusable as a retrieval summary, or None when it is acceptable."""
    if finish_reason == "length":
        return "truncated by max_tokens"
    if not MIN_SUMMARY_CHARS <= len(text) <= MAX_SUMMARY_CHARS:
        return f"length {len(text)} outside [{MIN_SUMMARY_CHARS}, {MAX_SUMMARY_CHARS}]"
    if _NOT_A_SUMMARY.match(text.lstrip("*# ")):
        return "reasoning/preamble instead of a summary"
    if text.count("**") >= 4 or text.count("\n- ") >= 3:
        return "markdown scaffolding instead of prose"
    return None


def summary_cache_key(content_sha256: str, prompt_version: str, model_name: str) -> str:
    return hashlib.sha256(f"{content_sha256}|{prompt_version}|{model_name}".encode()).hexdigest()[:32]


def document_excerpt(session: Session, version: RegulationVersion) -> str:
    """Front matter, the section outline and the first pages of body text — enough to say
    what the document is, bounded so a 4,000-page manual costs the same as a 40-page one."""
    sections = session.scalars(select(Section).where(Section.version_id == version.id).order_by(Section.ordinal)).all()
    front = "\n".join(s.content for s in sections if s.kind == "FRONT_MATTER")[:EXCERPT_FRONT_CHARS]
    outline: list[str] = []
    for s in sections:
        if s.kind == "FRONT_MATTER" or s.depth > 2 or not (s.title or s.section_number):
            continue
        outline.append(" ".join(x for x in (s.annex, s.section_number, s.title) if x))
        if len(outline) >= EXCERPT_OUTLINE_LINES:
            break
    body = "\n".join(s.content for s in sections if s.kind != "FRONT_MATTER" and s.content)[:EXCERPT_BODY_CHARS]
    parts = [p for p in (front, "\n".join(outline), body) if p.strip()]
    return "\n\n".join(parts)


def ensure_summary(
    session: Session,
    version: RegulationVersion,
    regulation: Regulation,
    llm: LLMProvider | None,
    *,
    allowed_data_classes: list[str],
    retry_failed: bool = False,
) -> DocumentSummary:
    """Return the cached summary row for this (artifact, prompt, model), generating it once."""
    from safety_assistant.persistence.models import SourceArtifact

    artifact = session.get(SourceArtifact, version.source_artifact_id)
    sha = artifact.sha256 if artifact else str(version.id)
    model = llm.model if llm else "none"
    key = summary_cache_key(sha, SUMMARY_PROMPT_VERSION, model)
    row = session.scalar(
        select(DocumentSummary).where(DocumentSummary.version_id == version.id, DocumentSummary.cache_key == key)
    )
    if row is not None and (row.status == "READY" or not retry_failed):
        return row
    if row is None:
        row = DocumentSummary(
            version_id=version.id,
            cache_key=key,
            content_sha256=sha,
            prompt_version=SUMMARY_PROMPT_VERSION,
            model_name=model,
            status="SKIPPED",
        )
        session.add(row)

    if llm is None:
        row.status, row.error = "SKIPPED", "no LLM provider configured"
    elif regulation.data_class not in allowed_data_classes:
        row.status, row.error = "SKIPPED", f"data class {regulation.data_class} not cleared for provider"
    else:
        try:
            resp = _generate_with_retry(
                llm,
                [
                    LLMMessage(role="system", content=SUMMARY_SYSTEM),
                    LLMMessage(
                        role="user",
                        content=summary_user_message(
                            document_context(regulation, version), document_excerpt(session, version)
                        ),
                    ),
                ],
            )
            text = resp.content.strip()
            problem = validate_summary(text, resp.finish_reason)
            if problem:
                raise ValueError(f"{problem} (model {resp.model})")
            row.status, row.summary, row.error = "READY", text, None
            row.usage = {**(resp.usage or {}), "routed_model": resp.model}
        except Exception as exc:  # noqa: BLE001 — one document's summary never blocks the corpus
            log.warning("document summary failed for %s %s: %s", regulation.regulation_key, version.version_label, exc)
            row.status, row.summary, row.error = "FAILED", None, f"{type(exc).__name__}: {exc}"[:500]
    session.flush()
    return row


def _generate_with_retry(
    llm: LLMProvider, messages: list[LLMMessage], *, delays: tuple[float, ...] = RETRY_DELAYS_S
) -> LLMResponse:
    for attempt, delay in enumerate((*delays, None)):
        try:
            return llm.generate(messages, max_tokens=SUMMARY_MAX_TOKENS)
        except LLMError as exc:
            if not exc.retryable or delay is None:
                raise
            log.info("summary generation retry %d after %s: %s", attempt + 1, type(exc).__name__, exc)
            time.sleep(delay)
    raise AssertionError("unreachable")
