"""One source through the lifecycle — idempotent, incremental, quarantining.

    DISCOVERED → DOWNLOADED → VALIDATED → PARSED → NORMALIZED → CHUNKED → INDEXED → VERIFIED → ACTIVE

Idempotency keys (CLAUDE.md §6):

    parse  = source_sha256 + parser_version + parser_config_hash
    chunk  = parsed_hash   + chunker_version + chunker_config_hash
    embed  = chunk_sha256  + model_version   + dimensions     (index/embed.py)
    index  = chunk_id      + index_schema_version

Re-running an unchanged source is a no-op (`SKIPPED_UNCHANGED`). A failed
validation/parse quarantines that version only; the stream continues.
Activation is atomic: the previous ACTIVE version of the same regulation is
marked SUPERSEDED in the same transaction that promotes the new one.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import subprocess
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from safety_assistant.config import Settings, get_settings
from safety_assistant.domain.regulations import VersionStatus, transition
from safety_assistant.ingestion.chunk import (
    CHUNKER_VERSION,
    CitationContext,
    chunk_document,
    chunker_config_hash,
)
from safety_assistant.ingestion.fetch.blobstore import BlobStore, blob_store_from_uri, sha256_bytes
from safety_assistant.ingestion.index.embed import (
    INDEX_SCHEMA_VERSION,
    EmbedStats,
    embed_version_chunks,
    snapshot_embeddings,
)
from safety_assistant.ingestion.normalize import (
    NormalizedDocument,
    normalize_generic,
    normalize_regulation,
    normalizer_config_hash,
    parse_cover,
)
from safety_assistant.ingestion.parse import DocumentParser, ParsedDocument, PyMuPDFParser
from safety_assistant.ingestion.sources.registry import SourceEntry, SourceRegistry, get_registry
from safety_assistant.ingestion.validation import ValidationError, validate_pdf_bytes
from safety_assistant.persistence.models import (
    Chunk,
    ChunkEmbedding,
    CrossReference,
    Figure,
    IngestionEvent,
    IngestionRun,
    Regulation,
    RegulationVersion,
    Section,
    SourceArtifact,
    Table,
)
from safety_assistant.providers.embeddings import EmbeddingProvider, get_embedding_provider

MAX_ATTEMPTS = 3


class QuarantineError(Exception):
    """Deterministic, non-retryable failure of this source (bad file, unparseable)."""


@dataclass
class IngestOutcome:
    run_id: uuid.UUID
    version_id: uuid.UUID | None
    status: str
    final_version_status: str | None
    stats: dict[str, Any]
    error: str | None = None


@dataclass
class _Ctx:
    session: Session
    run: IngestionRun
    settings: Settings
    parser: DocumentParser
    blobs: BlobStore
    embedder: EmbeddingProvider
    repo_root: pathlib.Path
    stats: dict[str, Any] = dataclasses.field(default_factory=dict)
    reuse_pool: dict[str, list[float]] = dataclasses.field(default_factory=dict)

    def event(
        self, to_status: str, message: str, *, from_status: str | None = None, level: str = "INFO", **payload: Any
    ) -> None:
        self.session.add(
            IngestionEvent(
                run_id=self.run.id,
                at=_now(),
                from_status=from_status,
                to_status=to_status,
                level=level,
                message=message,
                payload=payload or None,
            )
        )

    def advance(self, version: RegulationVersion, target: VersionStatus, message: str, **payload: Any) -> None:
        previous = version.status
        version.status = transition(previous, target).value
        self.event(version.status, message, from_status=previous, **payload)
        self.session.flush()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — not available in a container image
        return None


# --------------------------------------------------------------------------- public entry point


def ingest_source(
    session: Session,
    source_key: str,
    *,
    registry: SourceRegistry | None = None,
    parser: DocumentParser | None = None,
    blob_store: BlobStore | None = None,
    embedder: EmbeddingProvider | None = None,
    settings: Settings | None = None,
    repo_root: pathlib.Path | None = None,
    force: bool = False,
    activate: bool = True,
    max_pages: int | None = None,
) -> IngestOutcome:
    settings = settings or get_settings()
    registry = registry or get_registry()
    entry = registry.get(source_key)  # allowlist: KeyError for anything not registered
    ctx = _Ctx(
        session=session,
        run=IngestionRun(source_key=source_key, status="RUNNING", started_at=_now(), git_sha=_git_sha()),
        settings=settings,
        parser=parser or PyMuPDFParser(),
        blobs=blob_store or blob_store_from_uri(settings.artifact_store_uri),
        embedder=embedder or get_embedding_provider(),
        repo_root=repo_root or pathlib.Path.cwd(),
    )
    session.add(ctx.run)
    session.flush()

    version: RegulationVersion | None = None
    try:
        regulation = _upsert_regulation(session, entry)
        version, created = _get_or_create_version(session, regulation, entry)
        ctx.run.version_id = version.id
        attempts = int((version.metadata_ or {}).get("attempts", 0)) + 1
        version.metadata_ = {**(version.metadata_ or {}), "attempts": attempts}

        if not created and not force and _unchanged(version, entry, ctx):
            ctx.run.status = "SKIPPED_UNCHANGED"
            ctx.event(version.status, "source, parser and chunker configuration unchanged — nothing to do")
            return _finish(ctx, version)

        if version.status in (VersionStatus.QUARANTINED, VersionStatus.FAILED) and not force:
            if attempts > MAX_ATTEMPTS:
                ctx.run.status = "QUARANTINED"
                ctx.event(
                    version.status, f"attempt budget exhausted ({MAX_ATTEMPTS}); manual reset required", level="ERROR"
                )
                return _finish(ctx, version)
            ctx.advance(version, VersionStatus.DISCOVERED, "retrying from scratch", attempt=attempts)
        elif version.status != VersionStatus.DISCOVERED:
            # Explicit reprocess (force, or a re-run after a partial failure): the
            # version leaves the retrievable set until it is re-verified.
            previous = version.status
            version.status = transition(previous, VersionStatus.DISCOVERED, force=True).value
            ctx.event(version.status, "reprocessing from scratch", from_status=previous, forced=force)

        data = _download(ctx, version, entry)
        validated = _validate(ctx, version, entry, data)
        artifact = _artifact(ctx, entry, data, validated.sha256)
        version.source_artifact_id = artifact.id
        parsed = _parse(ctx, version, data, validated.sha256, max_pages)
        normalized = _normalize(ctx, version, entry, parsed)
        _chunk(ctx, version, entry, parsed, normalized)
        _index(ctx, version)
        _verify(ctx, version)
        if activate:
            _activate(ctx, version, regulation)
        ctx.run.status = "SUCCEEDED"
        return _finish(ctx, version)

    except QuarantineError as exc:
        ctx.run.status = "QUARANTINED"
        ctx.run.error = str(exc)
        if version is not None:
            _mark(ctx, version, VersionStatus.QUARANTINED, str(exc))
        return _finish(ctx, version)
    except Exception as exc:  # noqa: BLE001 — never let one source kill the stream
        ctx.run.status = "FAILED"
        ctx.run.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
        if version is not None:
            _mark(ctx, version, VersionStatus.FAILED, f"{type(exc).__name__}: {exc}")
        return _finish(ctx, version)


# --------------------------------------------------------------------------- stages


def _upsert_regulation(session: Session, e: SourceEntry) -> Regulation:
    reg = session.scalar(select(Regulation).where(Regulation.regulation_key == e.regulation_key))
    if reg is None:
        reg = Regulation(
            regulation_key=e.regulation_key,
            title=e.title,
            kind=e.kind,
            authority=e.authority,
            jurisdiction=e.jurisdiction,
            authority_level=e.authority_level,
            data_class=e.data_class,
            metadata_={"publisher": e.publisher, "license": e.license},
        )
        session.add(reg)
        session.flush()
    return reg


def _get_or_create_version(session: Session, reg: Regulation, e: SourceEntry) -> tuple[RegulationVersion, bool]:
    v = session.scalar(
        select(RegulationVersion).where(
            RegulationVersion.regulation_id == reg.id, RegulationVersion.version_label == e.version.label
        )
    )
    if v is not None:
        return v, False
    # An artifact row is required (NOT NULL); create/reuse it now from the registry hash.
    art = session.scalar(select(SourceArtifact).where(SourceArtifact.sha256 == e.sha256))
    if art is None:
        art = SourceArtifact(
            sha256=e.sha256,
            storage_uri="",
            filename=pathlib.Path(e.local_path).name,
            media_type=e.media_type,
            size_bytes=e.size_bytes,
            source_key=e.source_key,
            source_uri=e.source_uri,
            metadata_={"source_uri_status": e.source_uri_status},
        )
        session.add(art)
        session.flush()
    v = RegulationVersion(
        regulation_id=reg.id,
        source_artifact_id=art.id,
        version_label=e.version.label,
        series=e.version.series,
        revision=e.version.revision,
        published_at=e.version.published_at,
        valid_from=e.version.valid_from,
        valid_to=e.version.valid_to,
        status=VersionStatus.DISCOVERED.value,
        metadata_={"document_symbol": e.version.document_symbol, "source_key": e.source_key},
    )
    session.add(v)
    session.flush()
    return v, True


def _unchanged(version: RegulationVersion, e: SourceEntry, ctx: _Ctx) -> bool:
    if version.status != VersionStatus.ACTIVE:
        return False
    art = ctx.session.get(SourceArtifact, version.source_artifact_id)
    return (
        art is not None
        and art.sha256 == e.sha256
        and version.parser_version == ctx.parser.version
        and version.parser_config_hash == ctx.parser.config_hash()
        and version.chunker_version == CHUNKER_VERSION
        and version.chunker_config_hash == chunker_config_hash() + normalizer_config_hash()
        and version.index_schema_version == INDEX_SCHEMA_VERSION
    )


def _download(ctx: _Ctx, version: RegulationVersion, e: SourceEntry) -> bytes:
    path = ctx.repo_root / e.local_path
    if not path.is_file():
        raise QuarantineError(f"registered local copy missing: {e.local_path}")
    if path.stat().st_size > ctx.settings.ingest_max_file_bytes:
        raise QuarantineError(f"file exceeds ingest_max_file_bytes: {path.stat().st_size}")
    data = path.read_bytes()
    ctx.advance(
        version, VersionStatus.DOWNLOADED, f"read {len(data)} bytes from registered local copy", bytes=len(data)
    )
    return data


def _validate(ctx: _Ctx, version: RegulationVersion, e: SourceEntry, data: bytes) -> Any:
    try:
        v = validate_pdf_bytes(
            data,
            expected_sha256=e.sha256,
            expected_size=e.size_bytes,
            max_bytes=ctx.settings.ingest_max_file_bytes,
            max_pages=ctx.settings.ingest_max_pages,
        )
    except ValidationError as exc:
        raise QuarantineError(f"validation failed: {exc}") from exc
    ctx.advance(version, VersionStatus.VALIDATED, "sha256/size/magic/page-count checks passed", pages=v.page_count)
    return v


def _artifact(ctx: _Ctx, e: SourceEntry, data: bytes, sha: str) -> SourceArtifact:
    art = ctx.session.scalar(select(SourceArtifact).where(SourceArtifact.sha256 == sha))
    uri = ctx.blobs.put(data, suffix=".pdf")
    if art is None:
        art = SourceArtifact(
            sha256=sha,
            storage_uri=uri,
            filename=pathlib.Path(e.local_path).name,
            media_type="application/pdf",
            size_bytes=len(data),
            source_key=e.source_key,
            source_uri=e.source_uri,
            retrieved_at=_now(),
            metadata_={"source_uri_status": e.source_uri_status},
        )
        ctx.session.add(art)
    elif not art.storage_uri:
        art.storage_uri = uri
        art.retrieved_at = art.retrieved_at or _now()
    ctx.session.flush()
    return art


def _parse(ctx: _Ctx, version: RegulationVersion, data: bytes, sha: str, max_pages: int | None) -> ParsedDocument:
    started = _now()
    parsed = ctx.parser.parse(data, source_sha256=sha, max_pages=max_pages)
    report = parsed.report
    assert report is not None
    version.parser_name = ctx.parser.name
    version.parser_version = ctx.parser.version
    version.parser_config_hash = ctx.parser.config_hash()
    version.extraction_report = report.to_dict()
    ctx.stats["parse_seconds"] = (_now() - started).total_seconds()
    ctx.stats["pages"] = report.processed_page_count
    if report.status == "FAIL":
        raise QuarantineError(f"extraction QA FAIL: {len(report.failed_pages)} failed page(s)")
    ctx.advance(
        version,
        VersionStatus.PARSED,
        f"parsed {report.processed_page_count} pages, QA {report.status}",
        qa_status=report.status,
        failed_pages=report.failed_pages,
        tables=len(parsed.tables),
        figures=len(parsed.figures),
        route_summary=report.route_summary,
    )
    return parsed


def _normalize(ctx: _Ctx, version: RegulationVersion, e: SourceEntry, parsed: ParsedDocument) -> NormalizedDocument:
    if e.kind == "REGULATION":
        nd = normalize_regulation(parsed.pages)
        cover = parse_cover([p.text for p in parsed.pages])
        version.amendments = cover.to_json()
        meta = dict(version.metadata_ or {})
        meta["cover"] = {
            "document_symbol": cover.document_symbol,
            "revision": cover.revision,
            "document_date": cover.document_date.isoformat() if cover.document_date else None,
        }
        version.metadata_ = meta
        # Cross-check the human-reviewed registry against what the cover page says.
        if (
            cover.latest_entry_into_force
            and e.version.valid_from
            and cover.latest_entry_into_force != e.version.valid_from
        ):
            ctx.event(
                version.status,
                "registry valid_from disagrees with cover-page entry-into-force",
                level="WARNING",
                registry=e.version.valid_from.isoformat(),
                cover=cover.latest_entry_into_force.isoformat(),
            )
        if cover.document_symbol and e.version.document_symbol and cover.document_symbol != e.version.document_symbol:
            ctx.event(
                version.status,
                "registry document_symbol disagrees with cover page",
                level="WARNING",
                registry=e.version.document_symbol,
                cover=cover.document_symbol,
            )
    else:
        nd = normalize_generic(parsed.pages)
    ctx.stats["sections"] = len(nd.sections)
    ctx.stats["cross_references"] = len(nd.cross_references)
    ctx.advance(
        version,
        VersionStatus.NORMALIZED,
        f"{len(nd.sections)} sections, {len(nd.cross_references)} cross-references",
        parsed_hash=nd.parsed_hash,
        changed=(version.parsed_hash != nd.parsed_hash),
    )
    return nd


def _chunk(
    ctx: _Ctx, version: RegulationVersion, e: SourceEntry, parsed: ParsedDocument, nd: NormalizedDocument
) -> None:
    s = ctx.session
    cfg_hash = chunker_config_hash() + normalizer_config_hash()
    existing_chunks = s.scalar(select(func.count()).select_from(Chunk).where(Chunk.version_id == version.id)) or 0
    if (
        version.parsed_hash == nd.parsed_hash
        and version.chunker_version == CHUNKER_VERSION
        and version.chunker_config_hash == cfg_hash
        and existing_chunks > 0
    ):
        ctx.stats["chunks"] = existing_chunks
        ctx.stats["chunks_reused"] = True
        ctx.advance(
            version,
            VersionStatus.CHUNKED,
            "structure unchanged — existing sections/chunks kept",
            chunks=existing_chunks,
        )
        return

    # Rewrite structure for this version; keep this version's embeddings by content hash
    # so the index stage re-embeds only chunks whose text actually changed.
    ctx.reuse_pool = snapshot_embeddings(s, version, ctx.embedder)
    s.execute(delete(Chunk).where(Chunk.version_id == version.id))
    s.execute(delete(CrossReference).where(CrossReference.version_id == version.id))
    s.execute(delete(Table).where(Table.version_id == version.id))
    s.execute(delete(Figure).where(Figure.version_id == version.id))
    s.execute(delete(Section).where(Section.version_id == version.id))
    s.flush()

    section_ids: dict[str, uuid.UUID] = {}
    for ns in nd.sections:
        row = Section(
            version_id=version.id,
            parent_section_id=section_ids.get(ns.parent_path) if ns.parent_path else None,
            ordinal=ns.ordinal,
            path=ns.path,
            section_number=ns.section_number,
            annex=ns.annex,
            title=ns.title,
            kind=ns.kind,
            normative=ns.normative,
            depth=ns.depth,
            page_start=ns.page_start,
            page_end=ns.page_end,
            content=ns.content,
            content_sha256=ns.content_sha256,
        )
        s.add(row)
        s.flush()
        section_ids[ns.path] = row.id

    for xr in nd.cross_references:
        s.add(
            CrossReference(
                version_id=version.id,
                from_section_id=section_ids[xr.from_path],
                raw_text=xr.raw_text,
                target_path=xr.target_path,
                target_regulation_key=xr.target_regulation_key,
                resolved_section_id=section_ids.get(xr.target_path),
            )
        )

    page_to_section = _page_index(nd, section_ids)
    for t in parsed.tables:
        s.add(
            Table(
                version_id=version.id,
                section_id=page_to_section(t.page_number),
                page_number=t.page_number,
                table_index=t.table_index,
                headers=t.headers,
                rows=t.rows,
                bounding_box={"x0": t.bbox[0], "y0": t.bbox[1], "x1": t.bbox[2], "y1": t.bbox[3]},
                extraction_method=t.extraction_method,
                quality_score=t.quality_score,
                content_sha256=sha256_bytes(repr(t.rows).encode()),
            )
        )
    for f in parsed.figures:
        uri = ctx.blobs.put(f.image_bytes, suffix=f".{f.image_ext}")
        s.add(
            Figure(
                version_id=version.id,
                section_id=page_to_section(f.page_number),
                page_number=f.page_number,
                figure_index=f.figure_index,
                storage_uri=uri,
                image_sha256=sha256_bytes(f.image_bytes),
                bounding_box=({"x0": f.bbox[0], "y0": f.bbox[1], "x1": f.bbox[2], "y1": f.bbox[3]} if f.bbox else None),
                figure_type="embedded_image",
            )
        )

    drafts = chunk_document(nd, parsed.tables, CitationContext(e.regulation_key, e.version.label))
    for d in drafts:
        s.add(
            Chunk(
                id=d.deterministic_id(version.id),
                version_id=version.id,
                section_id=section_ids[d.section_path],
                ordinal=d.ordinal,
                chunk_type=d.chunk_type,
                content=d.content,
                token_count=d.token_count,
                page_start=d.page_start,
                page_end=d.page_end,
                citation_label=d.citation_label,
                chunk_sha256=d.chunk_sha256,
                metadata_=d.metadata or None,
            )
        )
    version.parsed_hash = nd.parsed_hash
    version.chunker_version = CHUNKER_VERSION
    version.chunker_config_hash = cfg_hash
    s.flush()
    ctx.stats["chunks"] = len(drafts)
    ctx.stats["tables"] = len(parsed.tables)
    ctx.stats["figures"] = len(parsed.figures)
    ctx.advance(version, VersionStatus.CHUNKED, f"{len(drafts)} chunks written", chunks=len(drafts))


def _page_index(nd: NormalizedDocument, ids: dict[str, uuid.UUID]) -> Any:
    starts = [(sec.page_start, sec.path) for sec in nd.sections if sec.kind != "FRONT_MATTER"]

    def lookup(page: int) -> uuid.UUID | None:
        best = None
        for start, path in starts:
            if start <= page:
                best = path
            else:
                break
        return ids.get(best) if best else None

    return lookup


def _index(ctx: _Ctx, version: RegulationVersion) -> None:
    started = _now()
    stats: EmbedStats = embed_version_chunks(ctx.session, version, ctx.embedder, reuse_pool=ctx.reuse_pool)
    version.index_schema_version = INDEX_SCHEMA_VERSION
    ctx.stats.update(
        embed_total=stats.total,
        embed_reused=stats.reused,
        embed_new=stats.embedded,
        embed_seconds=(_now() - started).total_seconds(),
        embedding_model=ctx.embedder.model_name,
    )
    ctx.advance(
        version,
        VersionStatus.INDEXED,
        f"{stats.embedded} embedded, {stats.reused} reused of {stats.total}",
        model=ctx.embedder.model_name,
        dimensions=ctx.embedder.dimensions,
    )


def _verify(ctx: _Ctx, version: RegulationVersion) -> None:
    s = ctx.session
    chunks = s.scalar(select(func.count()).select_from(Chunk).where(Chunk.version_id == version.id)) or 0
    embedded = (
        s.scalar(
            select(func.count())
            .select_from(ChunkEmbedding)
            .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
            .where(
                Chunk.version_id == version.id,
                ChunkEmbedding.model_name == ctx.embedder.model_name,
                ChunkEmbedding.model_version == ctx.embedder.model_version,
            )
        )
        or 0
    )
    unlabeled = (
        s.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.version_id == version.id, Chunk.citation_label == "")
        )
        or 0
    )
    problems = []
    if chunks == 0:
        problems.append("no chunks")
    if embedded != chunks:
        problems.append(f"embeddings {embedded} != chunks {chunks}")
    if unlabeled:
        problems.append(f"{unlabeled} chunks without citation label")
    if problems:
        raise QuarantineError("verification failed: " + "; ".join(problems))
    ctx.advance(version, VersionStatus.VERIFIED, "chunk/embedding/citation consistency verified", chunks=chunks)


def _activate(ctx: _Ctx, version: RegulationVersion, regulation: Regulation) -> None:
    s = ctx.session
    now = _now()
    others = s.scalars(
        select(RegulationVersion).where(
            RegulationVersion.regulation_id == regulation.id,
            RegulationVersion.id != version.id,
            RegulationVersion.status == VersionStatus.ACTIVE.value,
        )
    ).all()
    for old in others:
        newer = version.valid_from is None or old.valid_from is None or version.valid_from >= old.valid_from
        if newer:
            old.status = transition(old.status, VersionStatus.SUPERSEDED).value
            old.superseded_by_id = version.id
            old.superseded_at = now
            if old.valid_to is None and version.valid_from is not None:
                old.valid_to = version.valid_from
            ctx.event(
                old.status, f"superseded by {version.version_label}", from_status="ACTIVE", version_id=str(old.id)
            )
        else:
            # The incoming version is *older* than the active one: it becomes a closed historical window.
            if version.valid_to is None and old.valid_from is not None:
                version.valid_to = old.valid_from
            version.superseded_by_id = old.id
    version.activated_at = now
    ctx.advance(version, VersionStatus.ACTIVE, "activated atomically", superseded=[str(o.id) for o in others])
    if version.superseded_by_id is not None:
        version.status = transition(version.status, VersionStatus.SUPERSEDED).value
        ctx.event(version.status, "historical version: activated then marked superseded by the newer active text")
    if version.published_at:
        lag = (now.date() - version.published_at).days
        ctx.stats["freshness_lag_days_from_publication"] = lag
    s.flush()


def _mark(ctx: _Ctx, version: RegulationVersion, target: VersionStatus, message: str) -> None:
    try:
        ctx.advance(version, target, message)
    except Exception:  # noqa: BLE001 — a status that cannot legally transition still gets the event
        ctx.event(target.value, message, from_status=version.status, level="ERROR")
        version.status = target.value


def _finish(ctx: _Ctx, version: RegulationVersion | None) -> IngestOutcome:
    ctx.run.finished_at = _now()
    ctx.run.stats = ctx.stats or None
    ctx.run.attempt = int((version.metadata_ or {}).get("attempts", 1)) if version else 1
    ctx.session.commit()
    return IngestOutcome(
        run_id=ctx.run.id,
        version_id=version.id if version else None,
        status=ctx.run.status,
        final_version_status=version.status if version else None,
        stats=dict(ctx.stats),
        error=ctx.run.error,
    )
