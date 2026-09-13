"""Sparse leg: real BM25 (rank-bm25 BM25Okapi) over retrievable chunks.

The index is built once per process per corpus generation (count + newest
chunk timestamp of retrievable versions) and cached; scoring is O(corpus)
numpy work — measured, not assumed (see README "Measured results").

Scope is applied *after* scoring but *before* the top-k cut: BM25 scores
every chunk, then only chunks inside the scoped universe are ranked. That
keeps tight filters (one regulation, one historical version) from starving
the leg, which a pre-cut top-k over the whole corpus would do.

The tokenizer keeps ``*`` and ``_`` and joins decimal clause numbers so
identifiers like ``5.2.1.8``, ``*MAT_024`` and ``HIC15`` survive as single
terms — exact identifiers must route strongly toward lexical retrieval.

One index per representation: "content" tokenizes the chunk text, "sac_v1" the
summary-augmented ``retrieval_text`` (falling back to content where it is NULL).
"""

from __future__ import annotations

import datetime
import re
import threading
import uuid
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from safety_assistant.domain.regulations import RETRIEVABLE_CURRENT, RETRIEVABLE_HISTORICAL
from safety_assistant.persistence.models import Chunk, Regulation, RegulationVersion
from safety_assistant.retrieval.authz import DocumentRef, anonymous_allows
from safety_assistant.retrieval.filters import ScopeFilter, light_stem

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)+|[a-z0-9_*]+")


def tokenize(text: str) -> list[str]:
    return [light_stem(t) for t in _TOKEN_RE.findall(text.lower())]


@dataclass(frozen=True)
class VersionMeta:
    status: str
    valid_from: datetime.date | None
    valid_to: datetime.date | None
    regulation_key: str
    kind: str
    authority_level: str
    data_class: str
    doc: DocumentRef

    def in_scope(self, scope: ScopeFilter, today: datetime.date | None) -> bool:
        statuses = RETRIEVABLE_HISTORICAL if (scope.include_superseded or scope.as_of) else RETRIEVABLE_CURRENT
        if self.status not in {x.value for x in statuses} or self.data_class not in scope.data_classes:
            return False
        if not (scope.authz.allows(self.doc) if scope.authz is not None else anonymous_allows(self.doc)):
            return False
        d = scope.effective_date(today)
        if self.valid_from is not None and self.valid_from > d:
            return False
        if self.valid_to is not None and self.valid_to <= d:
            return False
        if scope.regulation_keys and self.regulation_key not in scope.regulation_keys:
            return False
        if scope.kinds and self.kind not in scope.kinds:
            return False
        return not (scope.authority_levels and self.authority_level not in scope.authority_levels)


@dataclass
class Bm25Index:
    generation: tuple[int, str]
    representation: str
    chunk_ids: list[uuid.UUID]
    version_ids: list[uuid.UUID]
    versions: dict[uuid.UUID, VersionMeta]
    bm25: BM25Okapi | None

    def scores(
        self, query: str, scope: ScopeFilter | None = None, today: datetime.date | None = None
    ) -> dict[uuid.UUID, float]:
        """Scored chunks (zero scores excluded), scope applied in memory via the version map."""
        if self.bm25 is None:
            return {}
        tokens = tokenize(query)
        if not tokens:
            return {}
        raw = self.bm25.get_scores(tokens)
        allowed_versions: set[uuid.UUID] | None = None
        if scope is not None:
            if scope.version_ids:
                allowed_versions = {uuid.UUID(v) for v in scope.version_ids}
            else:
                allowed_versions = {vid for vid, meta in self.versions.items() if meta.in_scope(scope, today)}
        out: dict[uuid.UUID, float] = {}
        for cid, vid, sc in zip(self.chunk_ids, self.version_ids, raw, strict=True):
            if sc > 0 and (allowed_versions is None or vid in allowed_versions):
                out[cid] = float(sc)
        return out


_lock = threading.Lock()
_cache: dict[str, Bm25Index] = {}


def _generation(session: Session) -> tuple[int, str]:
    stmt = (
        select(func.count(Chunk.id), func.coalesce(func.max(Chunk.created_at), datetime.datetime.min))
        .join(RegulationVersion, RegulationVersion.id == Chunk.version_id)
        .where(RegulationVersion.status.in_([s.value for s in RETRIEVABLE_HISTORICAL]))
    )
    count, newest = session.execute(stmt).one()
    return int(count), str(newest)


def build_index(session: Session, representation: str = "content") -> Bm25Index:
    gen = _generation(session)
    text = func.coalesce(Chunk.retrieval_text, Chunk.content) if representation == "sac_v1" else Chunk.content
    stmt = (
        select(Chunk.id, text)
        .join(RegulationVersion, RegulationVersion.id == Chunk.version_id)
        .where(RegulationVersion.status.in_([s.value for s in RETRIEVABLE_HISTORICAL]))
    )
    rows: list[tuple[uuid.UUID, str, uuid.UUID]] = [
        (cid, txt, vid) for cid, txt, vid in session.execute(stmt.add_columns(Chunk.version_id)).all()
    ]
    if representation == "sac_v2":
        from safety_assistant.contextualization.context_builder import compact_prefixes

        prefixes = compact_prefixes(session)
        rows = [(cid, f"{prefixes.get(vid, '')}\n{txt}", vid) for cid, txt, vid in rows]
    versions = {
        v.id: VersionMeta(
            v.status,
            v.valid_from,
            v.valid_to,
            r.regulation_key,
            r.kind,
            r.authority_level,
            r.data_class,
            DocumentRef(r.id, r.scope, r.organization_id, r.workspace_id, r.owner_user_id, r.archived_at is not None),
        )
        for v, r in session.execute(
            select(RegulationVersion, Regulation).join(Regulation, Regulation.id == RegulationVersion.regulation_id)
        ).all()
    }
    if not rows:
        return Bm25Index(
            generation=gen, representation=representation, chunk_ids=[], version_ids=[], versions=versions, bm25=None
        )
    return Bm25Index(
        generation=gen,
        representation=representation,
        chunk_ids=[r[0] for r in rows],
        version_ids=[r[2] for r in rows],
        versions=versions,
        bm25=BM25Okapi([tokenize(r[1]) for r in rows]),
    )


def get_index(session: Session, representation: str = "content") -> Bm25Index:
    """Process-wide cache per representation, rebuilt when the retrievable corpus changes."""
    gen = _generation(session)
    with _lock:
        idx = _cache.get(representation)
        if idx is None or idx.generation != gen:
            idx = _cache[representation] = build_index(session, representation)
        return idx


def invalidate_cache() -> None:
    with _lock:
        _cache.clear()


def sparse_search(
    session: Session,
    query: str,
    scope: ScopeFilter,
    *,
    top_k: int,
    index: Bm25Index | None = None,
    today: datetime.date | None = None,
    representation: str = "content",
) -> list[tuple[uuid.UUID, float]]:
    """Ranked (chunk_id, bm25_score), scoped, zero-score chunks excluded."""
    idx = index or get_index(session, representation)
    scored = idx.scores(query, scope, today)
    return sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_k]
