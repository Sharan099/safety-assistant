"""Summary-augmented chunking (SAC): the retrieval representation of a chunk.

    retrieval_text = deterministic document identity block
                   + generated document summary (when one exists)
                   + the unchanged chunk content

Only ``retrieval_text`` is indexed by the "sac_v1" representation. ``Chunk.content`` is never
rewritten and remains the only text served as evidence — the summary is retrieval metadata.

The identity block is built from columns that ingestion extracted deterministically
(registry, cover page); unknown values are simply omitted, never guessed.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from safety_assistant.persistence.models import Chunk, ChunkEmbedding, DocumentSummary, Regulation, RegulationVersion

SAC_REPRESENTATION = "sac_v1"
# Compact variant: one identity line + the summary's first sentence + content. Built for embedders
# with a short input window (fastembed's MiniLM truncates at 128 tokens, so the full sac_v1 prefix
# alone fills it) and for BM25 with far fewer repeated document tokens per chunk.
SAC_COMPACT_REPRESENTATION = "sac_v2"
BASELINE_REPRESENTATION = "content"
CONTEXT_BUILDER_VERSION = "1"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def document_context(regulation: Regulation, version: RegulationVersion) -> str:
    """Structured, deterministic document identity. Every line is a known column value."""
    fields: list[tuple[str, object]] = [
        ("Regulation", regulation.regulation_key.replace("-", " ")),
        ("Title", regulation.title),
        ("Document type", regulation.kind),
        ("Authority", regulation.authority),
        ("Jurisdiction", regulation.jurisdiction),
        ("Version", version.version_label),
        ("Series", version.series),
        ("Revision", version.revision),
        ("Published", version.published_at),
        ("Effective from", version.valid_from),
        ("Effective to", version.valid_to),
    ]
    lines = ["SOURCE DOCUMENT"] + [f"{k}: {v}" for k, v in fields if v not in (None, "")]
    return "\n".join(lines)


def compact_context(regulation: Regulation, version: RegulationVersion, summary: str | None) -> str:
    """One line of document identity: "UN R94 — <title> (Rev.4 (04 series)). <first summary sentence>"."""
    head = f"{regulation.regulation_key.replace('-', ' ')} — {regulation.title} ({version.version_label})."
    if summary:
        first = _SENTENCE_END.split(summary.strip(), maxsplit=1)[0].strip()
        if first:
            head = f"{head} {first}"
    return head


def build_compact_text(prefix: str, content: str) -> str:
    return f"{prefix}\n{content}"


def compact_prefixes(session: Session, version_ids: list[uuid.UUID] | None = None) -> dict[uuid.UUID, str]:
    """version_id → compact identity line, using each version's READY summary when one exists."""
    stmt = select(RegulationVersion, Regulation).join(Regulation, Regulation.id == RegulationVersion.regulation_id)
    if version_ids is not None:
        stmt = stmt.where(RegulationVersion.id.in_(version_ids))
    ready = {
        d.version_id: d.summary
        for d in session.scalars(select(DocumentSummary).where(DocumentSummary.status == "READY")).all()
    }
    return {v.id: compact_context(r, v, ready.get(v.id)) for v, r in session.execute(stmt).all()}


def build_retrieval_text(context: str, summary: str | None, content: str) -> str:
    parts = [context]
    if summary:
        parts.append(f"DOCUMENT SUMMARY\n{summary.strip()}")
    parts.append(content)
    return "\n\n".join(parts)


def sac_config_hash(prompt_version: str, model_name: str) -> str:
    cfg = {"builder": CONTEXT_BUILDER_VERSION, "prompt": prompt_version, "model": model_name}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class ContextualizeStats:
    chunks: int
    summary_status: str  # READY | FAILED | SKIPPED
    reused: bool  # retrieval_text already current for this configuration


def contextualize_version(
    session: Session,
    version: RegulationVersion,
    regulation: Regulation,
    summary: DocumentSummary | None,
    *,
    force: bool = False,
) -> ContextualizeStats:
    """Write ``retrieval_text`` for every chunk of the version. Idempotent: the configuration
    hash is stored on the version and unchanged chunks are left alone. When the text is
    rewritten, the version's sac_v1 vectors are dropped so the index stage re-embeds them —
    a vector of stale retrieval text must never survive a summary change."""
    status = summary.status if summary else "SKIPPED"
    text = summary.summary if summary and summary.status == "READY" else None
    key = sac_config_hash(summary.prompt_version if summary else "-", summary.model_name if summary else "-")
    meta = version.metadata_ or {}
    current = meta.get("sac") == {"config": key, "summary_status": status}
    chunks = session.scalars(select(Chunk).where(Chunk.version_id == version.id)).all()
    if current and not force and all(c.retrieval_text for c in chunks):
        return ContextualizeStats(chunks=len(chunks), summary_status=status, reused=True)
    context = document_context(regulation, version)
    for c in chunks:
        c.retrieval_text = build_retrieval_text(context, text, c.content)
    session.execute(
        delete(ChunkEmbedding).where(
            ChunkEmbedding.chunk_id.in_([c.id for c in chunks]),
            ChunkEmbedding.representation != BASELINE_REPRESENTATION,
        )
    )
    version.metadata_ = {**meta, "sac": {"config": key, "summary_status": status}}
    session.flush()
    return ContextualizeStats(chunks=len(chunks), summary_status=status, reused=False)
