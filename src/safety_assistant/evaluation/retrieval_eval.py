"""Retrieval evaluation runner — measures each pipeline leg independently.

Ground truth is expressed as section paths; the relevant chunk-id set is
resolved from the live corpus at run time so it survives re-chunking.
Every result records dataset version, git SHA, corpus fingerprint,
parser/chunker/embedding/reranker identities and the retrieval config.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from safety_assistant.evaluation.dataset import GoldCase, GoldDataset
from safety_assistant.evaluation.metrics import (
    first_rank,
    hit_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from safety_assistant.persistence.models import Chunk, Regulation, RegulationVersion, Section
from safety_assistant.providers.embeddings import EmbeddingProvider
from safety_assistant.retrieval import RetrievalConfig, RetrievalService, ScopeFilter
from safety_assistant.retrieval.sparse import get_index

KS = (5, 10, 20)

LEGS: dict[str, dict[str, Any]] = {
    "dense": dict(use_sparse=False, use_exact=False, use_reranker=False, expand_parents=False, expand_cross_refs=False),
    "sparse": dict(use_dense=False, use_exact=False, use_reranker=False, expand_parents=False, expand_cross_refs=False),
    "hybrid_rrf": dict(use_exact=False, use_reranker=False, expand_parents=False, expand_cross_refs=False),
    "hybrid_rrf_rerank": dict(use_exact=False, expand_parents=False, expand_cross_refs=False),
    "full": {},
}


@dataclass
class CaseResult:
    case_id: str
    query_type: str
    answerability: str
    n_relevant_chunks: int
    ranked: list[str]
    first_relevant_rank: int | None
    regulation_hit_at_5: float | None
    metrics: dict[str, float | None]
    latency_ms: float
    top_citations: list[str]
    reranked: bool = True  # False when an adaptive policy skipped the reranker for this query


@dataclass
class LegReport:
    leg: str
    config: dict[str, Any]
    cases: list[CaseResult]
    aggregate: dict[str, float | None]
    by_slice: dict[str, dict[str, float | None]]


@dataclass
class EvalReport:
    dataset_version: str
    n_cases: int
    n_cases_with_section_truth: int
    git_sha: str | None
    timestamp: str
    corpus: dict[str, Any]
    versions: dict[str, Any]
    legs: list[LegReport] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def corpus_fingerprint(session: Session) -> dict[str, Any]:
    active = session.execute(
        select(Regulation.regulation_key, RegulationVersion.version_label, RegulationVersion.status,
               RegulationVersion.parser_version, RegulationVersion.chunker_version, RegulationVersion.parsed_hash)
        .join(RegulationVersion, RegulationVersion.regulation_id == Regulation.id)
        .order_by(Regulation.regulation_key)
    ).all()  # fmt: skip
    return {
        "chunks": session.scalar(select(func.count(Chunk.id))),
        "versions": [
            dict(zip(("regulation", "version", "status", "parser", "chunker", "parsed_hash"), r, strict=True))
            for r in active
        ],
    }


def relevant_chunk_ids(session: Session, case: GoldCase) -> set[uuid.UUID]:
    """Chunks whose section path / merged paths satisfy the case's section truth,
    restricted to the case's regulation(s)."""
    if not case.has_section_truth:
        return set()
    stmt = (
        select(Chunk.id, Section.path, Chunk.metadata_)
        .join(Section, Section.id == Chunk.section_id)
        .join(RegulationVersion, RegulationVersion.id == Chunk.version_id)
        .join(Regulation, Regulation.id == RegulationVersion.regulation_id)
    )
    if case.regulation_keys:
        stmt = stmt.where(Regulation.regulation_key.in_(case.regulation_keys))
    out: set[uuid.UUID] = set()
    for cid, path, meta in session.execute(stmt).all():
        merged = (meta or {}).get("merged_paths") if meta else None
        if case.section_matches(path, merged):
            out.add(cid)
    return out


def evaluate_leg(
    session: Session, dataset: GoldDataset, leg: str, *, service: RetrievalService, k_eval: int = 20
) -> LegReport:
    cases: list[CaseResult] = []
    for case in dataset.cases:
        relevant = relevant_chunk_ids(session, case)
        scope = ScopeFilter(as_of=case.as_of_date) if case.as_of_date else None
        t0 = time.perf_counter()
        result = service.search(session, case.query, scope=scope, k=k_eval)
        latency = (time.perf_counter() - t0) * 1000
        ranked = [e.chunk_id for e in result.bundle.evidence]
        reg_hit = None
        if case.regulation_keys:
            top_regs = {e.regulation_key for e in result.bundle.evidence[:5]}
            reg_hit = 1.0 if top_regs & case.regulation_keys else 0.0
        metrics: dict[str, float | None] = {}
        for k in KS:
            metrics[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
            metrics[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
            metrics[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
        metrics["mrr"] = reciprocal_rank(ranked, relevant)
        cases.append(
            CaseResult(
                case_id=case.case_id,
                query_type=case.query_type,
                answerability=case.answerability,
                n_relevant_chunks=len(relevant),
                ranked=[str(c) for c in ranked],
                first_relevant_rank=first_rank(ranked, relevant),
                regulation_hit_at_5=reg_hit,
                metrics=metrics,
                latency_ms=round(latency, 1),
                top_citations=[e.citation_label for e in result.bundle.evidence[:3]],
                reranked=not str(result.versions.get("reranker", "")).startswith("skipped"),
            )  # fmt: skip
        )
    return LegReport(
        leg=leg,
        config=dataclasses.asdict(service.config),
        cases=cases,
        aggregate=_aggregate(cases),
        by_slice={
            qt: _aggregate([c for c in cases if c.query_type == qt]) for qt in sorted({c.query_type for c in cases})
        },
    )


def _aggregate(cases: list[CaseResult]) -> dict[str, float | None]:
    if not cases:
        return {}
    names = [f"{m}@{k}" for k in KS for m in ("recall", "precision", "hit", "ndcg")] + ["mrr"]
    agg: dict[str, float | None] = {n: mean([c.metrics[n] for c in cases]) for n in names}
    agg["regulation_hit@5"] = mean([c.regulation_hit_at_5 for c in cases])
    agg["n"] = float(len(cases))
    agg["n_with_section_truth"] = float(sum(1 for c in cases if c.n_relevant_chunks))
    agg["rerank_rate"] = sum(1 for c in cases if c.reranked) / len(cases)
    lat = sorted(c.latency_ms for c in cases)
    if lat:
        agg["latency_p50_ms"] = lat[len(lat) // 2]
        agg["latency_p95_ms"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    return agg


def run_evaluation(
    session: Session,
    dataset: GoldDataset,
    *,
    legs: list[str] | None = None,
    base_config: RetrievalConfig | None = None,
    embedder: EmbeddingProvider | None = None,
) -> EvalReport:
    base = base_config or RetrievalConfig()
    bm25 = get_index(session)
    report = EvalReport(
        dataset_version=dataset.dataset_version,
        n_cases=len(dataset.cases),
        n_cases_with_section_truth=sum(1 for c in dataset.cases if c.has_section_truth),
        git_sha=_git_sha(),
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        corpus=corpus_fingerprint(session),
        versions={},
    )
    for leg in legs or list(LEGS):
        cfg = dataclasses.replace(base, **LEGS[leg])
        service = RetrievalService(config=cfg, bm25_index=bm25, embedder=embedder)
        report.legs.append(evaluate_leg(session, dataset, leg, service=service))
        report.versions.setdefault("embedding_model", service.embedder.model_name if cfg.use_dense else None)
        if cfg.use_reranker and service.reranker:
            report.versions.setdefault("reranker", f"{service.reranker.model_name}:{service.reranker.model_version}")
    return report


def write_report(report: EvalReport, out_dir: pathlib.Path, name: str = "retrieval") -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.timestamp.replace(":", "").replace("-", "")
    path = out_dir / f"{name}_{report.dataset_version}_{stamp}.json"
    path.write_text(json.dumps(report.to_json(), indent=2, default=str), encoding="utf-8")
    latest = out_dir / f"{name}_{report.dataset_version}_latest.json"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def format_summary(report: EvalReport) -> str:
    cols = ("recall@5", "recall@10", "recall@20", "precision@5", "hit@5", "mrr", "ndcg@10", "regulation_hit@5")
    heads = ("R@5", "R@10", "R@20", "P@5", "Hit@5", "MRR", "nDCG@10", "RegHit@5")
    lines = [
        f"dataset {report.dataset_version}: {report.n_cases} cases "
        f"({report.n_cases_with_section_truth} with section-level truth)",
        f"git {report.git_sha}  corpus chunks {report.corpus['chunks']}  {report.timestamp}",
        "",
        f"{'leg':20s} " + " ".join(f"{h:>8s}" for h in heads) + f" {'p50ms':>7s} {'p95ms':>7s}",
    ]
    for leg in report.legs:
        a = leg.aggregate
        cells = " ".join("     n/a" if a[c] is None else f"{a[c]:8.3f}" for c in cols)
        lines.append(f"{leg.leg:20s} {cells} {a.get('latency_p50_ms') or 0:7.0f} {a.get('latency_p95_ms') or 0:7.0f}")
    return "\n".join(lines)
