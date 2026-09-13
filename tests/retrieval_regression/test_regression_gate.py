"""Deterministic retrieval regression against the ingested corpus (CLAUDE.md §11).

Runs only when the development corpus (real fastembed vectors, all sources
ACTIVE) is reachable — never against the hashing test profile. Each stable
case must keep its regulation in the top 5 and its section within the top 10;
thresholds are the *measured* floor from evals/results, not aspirations.
"""

from __future__ import annotations

import os
import pathlib

import pytest

CORPUS_URL = os.environ.get(
    "CORPUS_DATABASE_URL", "postgresql+psycopg://passive_safety:change_me@localhost:5433/safety_assistant"
)
ROOT = pathlib.Path(__file__).resolve().parents[2]

STABLE_CASES = [  # case_id, min regulation rank (<=5), section must appear within top-k
    ("r94-002", 10), ("r94-004", 10), ("r94-006", 10), ("r94-008", 10), ("r94-013", 10),
    ("r95-002", 10), ("r95-003", 10), ("r95-004", 10), ("r95-005", 10),
    ("r16-001", 10), ("r16-002", 10), ("r16-003", 10), ("r16-004", 10), ("r16-005", 10), ("r16-006", 10),
    ("r129-001", 10), ("r129-002", 10), ("r129-004", 10), ("r129-005", 10), ("r129-006", 10), ("r129-007", 10),
    ("adv-001", 10), ("adv-002", 10),
]  # fmt: skip
MIN_MRR_FULL = 0.60  # measured 0.664 on 2026-09-11 (evals/results); fails on a real regression, not on noise


@pytest.fixture(scope="module")
def corpus():  # type: ignore[no-untyped-def]
    try:
        import psycopg

        with psycopg.connect(CORPUS_URL.replace("postgresql+psycopg://", "postgresql://"), connect_timeout=3) as c:
            n = c.execute("SELECT count(*) FROM regulation_versions WHERE status='ACTIVE'").fetchone()[0]
            model = c.execute("SELECT DISTINCT model_name FROM chunk_embeddings LIMIT 2").fetchall()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"corpus database not reachable: {exc}")
    if n < 4 or not model or "hashing" in str(model):
        pytest.skip("real corpus with fastembed vectors not present")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from safety_assistant.config.settings import Settings
    from safety_assistant.providers.embeddings import build_embedding_provider
    from safety_assistant.retrieval import RetrievalService

    embedder = build_embedding_provider(Settings(app_env="development", embedding_provider="fastembed"))
    engine = create_engine(CORPUS_URL)
    with Session(engine) as session:
        yield session, RetrievalService(embedder=embedder)


def test_stable_cases_keep_regulation_and_section_in_range(corpus) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.evaluation import load_dataset, relevant_chunk_ids
    from safety_assistant.retrieval import ScopeFilter

    session, svc = corpus
    ds = load_dataset(ROOT / "evals" / "datasets" / "regulatory_v1.yaml")
    by_id = {c.case_id: c for c in ds.cases}
    failures = []
    for case_id, k in STABLE_CASES:
        case = by_id[case_id]
        scope = ScopeFilter(as_of=case.as_of_date) if case.as_of_date else None
        r = svc.search(session, case.query, scope=scope, k=k)
        regs = [e.regulation_key for e in r.bundle.evidence[:5]]
        if not set(regs) & case.regulation_keys:
            failures.append(f"{case_id}: regulation {case.regulation_keys} not in top-5 {regs}")
            continue
        rel = relevant_chunk_ids(session, case)
        if rel and not {e.chunk_id for e in r.bundle.evidence[:k]} & rel:
            failures.append(
                f"{case_id}: section {case.expected_section_paths or case.expected_section_prefixes} not in top-{k}"
            )
    assert not failures, "\n".join(failures)


def test_historical_scope_never_returns_versions_outside_validity(corpus) -> None:  # type: ignore[no-untyped-def]
    import datetime

    from safety_assistant.retrieval import ScopeFilter

    session, svc = corpus
    r = svc.search(
        session,
        "tibia index limit",
        scope=ScopeFilter(as_of=datetime.date(2015, 1, 1), regulation_keys=("UN-R94",)),
        k=5,
    )
    assert r.bundle.evidence == []  # only Rev.4 (in force 2021-06-09) is ingested


def test_full_pipeline_mrr_floor(corpus) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.evaluation import load_dataset, run_evaluation

    session, svc = corpus
    ds = load_dataset(ROOT / "evals" / "datasets" / "regulatory_v1.yaml")
    report = run_evaluation(session, ds, legs=["full"], embedder=svc.embedder)
    mrr = report.legs[0].aggregate["mrr"]
    assert mrr is not None and mrr >= MIN_MRR_FULL, f"full-pipeline MRR {mrr:.3f} below measured floor {MIN_MRR_FULL}"


# regulatory_v2 (262 cases incl. 200 AUTO_GROUNDED): heuristic reranker measured 0.713 (2026-09-12);
# the cross-encoder default measured 0.808 — the floor guards the configuration the test profile runs.
MIN_MRR_FULL_V2 = 0.68


def test_full_pipeline_mrr_floor_regulatory_v2(corpus) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.evaluation import load_dataset, run_evaluation

    session, svc = corpus
    ds = load_dataset(ROOT / "evals" / "datasets" / "regulatory_v2.yaml")
    report = run_evaluation(session, ds, legs=["full"], embedder=svc.embedder)
    agg = report.legs[0].aggregate
    assert agg["mrr"] is not None and agg["mrr"] >= MIN_MRR_FULL_V2, f"v2 MRR {agg['mrr']:.3f} < {MIN_MRR_FULL_V2}"
    assert agg["recall@10"] >= 0.90, f"v2 R@10 {agg['recall@10']:.3f} < 0.90"
