"""Build a summary-augmented index (sac_v2 compact by default, sac_v1 full) for an existing corpus.

Runs per retrievable version, committing after each one, so an interrupted run resumes
where it stopped and a second run over a finished corpus is a no-op:

    summary   cached by (artifact sha256, prompt version, model) — generated once;
    text      `retrieval_text` written only when the SAC configuration changed;
    vectors   only chunks without a sac_v1 embedding are embedded.

The baseline `content` index is never touched; switching the query side is a separate
setting (`RETRIEVAL_REPRESENTATION`) taken after the A/B evaluation.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from safety_assistant.contextualization.context_builder import (
    SAC_COMPACT_REPRESENTATION,
    compact_prefixes,
    contextualize_version,
)
from safety_assistant.contextualization.document_summary import ensure_summary
from safety_assistant.domain.regulations import RETRIEVABLE_HISTORICAL
from safety_assistant.ingestion.index.embed import embed_version_chunks
from safety_assistant.persistence.models import Chunk, ChunkEmbedding, Regulation, RegulationVersion
from safety_assistant.providers.embeddings import EmbeddingProvider
from safety_assistant.providers.llm import LLMProvider
from safety_assistant.retrieval.sparse import invalidate_cache

log = logging.getLogger(__name__)


@dataclasses.dataclass
class VersionOutcome:
    regulation_key: str
    version_label: str
    chunks: int
    summary_status: str
    embedded: int
    reused: int
    seconds: float
    error: str | None = None


def reindex_sac(
    session_factory: Callable[[], Session],
    embedder: EmbeddingProvider,
    llm: LLMProvider | None,
    *,
    allowed_data_classes: list[str],
    retry_failed_summaries: bool = False,
    regulation_keys: list[str] | None = None,
    progress: Callable[[str], None] = log.info,
    representation: str = SAC_COMPACT_REPRESENTATION,
) -> list[VersionOutcome]:
    with session_factory() as s:
        stmt = (
            select(RegulationVersion.id)
            .join(Regulation, Regulation.id == RegulationVersion.regulation_id)
            .where(RegulationVersion.status.in_([x.value for x in RETRIEVABLE_HISTORICAL]))
            .order_by(Regulation.regulation_key, RegulationVersion.version_label)
        )
        if regulation_keys:
            stmt = stmt.where(Regulation.regulation_key.in_(regulation_keys))
        version_ids = list(s.scalars(stmt).all())

    outcomes: list[VersionOutcome] = []
    for n, vid in enumerate(version_ids, 1):
        t0 = time.perf_counter()
        with session_factory() as s:
            version = s.get(RegulationVersion, vid)
            regulation = s.get(Regulation, version.regulation_id) if version else None
            assert version is not None and regulation is not None
            key, label = regulation.regulation_key, version.version_label
            try:
                summary = ensure_summary(
                    s,
                    version,
                    regulation,
                    llm,
                    allowed_data_classes=allowed_data_classes,
                    retry_failed=retry_failed_summaries,
                )
                ctx = contextualize_version(s, version, regulation, summary)
                prefix = None
                if representation == SAC_COMPACT_REPRESENTATION:
                    prefix = compact_prefixes(s, [version.id]).get(version.id)
                stats = embed_version_chunks(s, version, embedder, representation=representation, prefix=prefix)
                s.commit()
                out = VersionOutcome(key, label, ctx.chunks, ctx.summary_status, stats.embedded, stats.reused, _s(t0))
            except Exception as exc:  # noqa: BLE001 — one version's failure must not stop the corpus
                s.rollback()
                log.exception("reindex failed for %s %s", key, label)
                out = VersionOutcome(key, label, 0, "?", 0, 0, _s(t0), error=f"{type(exc).__name__}: {exc}")
        outcomes.append(out)
        progress(
            f"[{n}/{len(version_ids)}] {out.regulation_key} {out.version_label}: {out.chunks} chunks, "
            f"summary {out.summary_status}, embedded {out.embedded} (+{out.reused} reused) in {out.seconds:.1f}s"
            + (f" ERROR {out.error}" if out.error else "")
        )
    invalidate_cache()
    return outcomes


def sac_coverage(
    session: Session, embedder: EmbeddingProvider, representation: str = SAC_COMPACT_REPRESENTATION
) -> dict[str, int]:
    """Chunks in retrievable versions vs chunks with a sac_v1 embedding — the readiness check
    before switching `RETRIEVAL_REPRESENTATION`."""
    retrievable = select(RegulationVersion.id).where(
        RegulationVersion.status.in_([x.value for x in RETRIEVABLE_HISTORICAL])
    )
    total = session.scalar(select(func.count(Chunk.id)).where(Chunk.version_id.in_(retrievable))) or 0
    covered = (
        session.scalar(
            select(func.count(ChunkEmbedding.id))
            .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
            .where(
                Chunk.version_id.in_(retrievable),
                ChunkEmbedding.representation == representation,
                ChunkEmbedding.model_name == embedder.model_name,
                ChunkEmbedding.model_version == embedder.model_version,
            )
        )
        or 0
    )
    return {"chunks": int(total), "sac_embedded": int(covered)}


def _s(t0: float) -> float:
    return round(time.perf_counter() - t0, 1)
