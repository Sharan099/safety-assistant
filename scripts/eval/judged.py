"""End-to-end evaluation: deterministic answer metrics + LLM-judged metrics (RAGAS, DeepEval).

    uv sync --extra eval
    uv run python scripts/eval/judged.py --dataset evals/datasets/regulatory_v2.yaml \
        [--slices numeric_threshold definition ...] [--limit 120] [--ragas] [--deepeval] [--deepeval-limit 40]

Stage 1 runs the real pipeline (retrieval → gate → generation → validation) per case with the
configured LLM and scores the deterministic metrics (`safety_assistant.evaluation.generation_eval`).
Answers are cached under evals/cache/answers/<fingerprint>_<case>.json.

Stage 2 (optional) judges the same records with
- RAGAS   faithfulness, answer_relevancy, context_precision, context_recall
- DeepEval FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
using the *same* OpenAI-compatible gateway as the product (free-tier friendly: sequential, cached per
record+metric under evals/cache/judge/). Judged scores are reported per slice next to the deterministic
ones; the judge model is recorded. Unanswerable/adversarial cases are excluded from answer-quality judges
(their metric is refusal accuracy / injection resistance).

Every number in the report carries dataset version, git SHA, pipeline fingerprint, judge model.
"""

# mypy: ignore-errors
# (integrates two untyped judge frameworks behind the optional `eval` extra)
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any

from sqlalchemy.orm import Session

from safety_assistant.config import get_settings
from safety_assistant.evaluation.dataset import load_dataset
from safety_assistant.evaluation.generation_eval import (
    CaseRecord,
    cost_estimate,
    pipeline_fingerprint,
    report,
    run_case,
)
from safety_assistant.evaluation.metrics import mean
from safety_assistant.generation.service import AnswerService
from safety_assistant.persistence import get_engine

JUDGE_CACHE = pathlib.Path("evals/cache/judge")
# Judges must never phone home with regulatory text or usage data.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "0")


def _judgeable(r: CaseRecord) -> bool:
    return r.answerability == "answerable" and r.query_type != "adversarial_injection" and bool(r.answer)


def _cache_key(kind: str, rec: CaseRecord, judge_model: str) -> pathlib.Path:
    h = hashlib.sha256(
        f"{kind}|{judge_model}|{rec.case_id}|{rec.answer}|{'|'.join(rec.contexts)}".encode()
    ).hexdigest()[:24]
    return JUDGE_CACHE / f"{kind}_{h}.json"


# ------------------------------------------------------------------ RAGAS


class _ProductEmbeddings:
    """LangChain-shaped adapter over the product's embedding provider (same vectors as retrieval)."""

    def __init__(self) -> None:
        from safety_assistant.providers.embeddings import get_embedding_provider

        self._p = get_embedding_provider()
        self.model = getattr(self._p, "model_name", "product-embeddings")

    def embed_query(self, text: str) -> list[float]:
        return list(self._p.embed_query(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self._p.embed_documents(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _ragas_scores(records: list[CaseRecord], judge_model: str) -> dict[str, dict[str, float | None]]:
    from langchain_openai import ChatOpenAI
    from ragas import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    s = get_settings()
    llm = LangchainLLMWrapper(
        ChatOpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key or "x", model=judge_model, temperature=0, timeout=120)
    )
    emb = LangchainEmbeddingsWrapper(_ProductEmbeddings())  # type: ignore[arg-type]
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }
    out: dict[str, dict[str, float | None]] = {}
    for i, rec in enumerate(records, 1):
        cache = _cache_key("ragas", rec, judge_model)
        if cache.exists():
            out[rec.case_id] = json.loads(cache.read_text(encoding="utf-8"))
            continue
        reference = " ".join(rec.key_facts) or (rec.answer or "")
        sample = SingleTurnSample(
            user_input=rec.query, response=rec.answer or "", retrieved_contexts=rec.contexts[:8], reference=reference
        )
        scores: dict[str, float | None] = {}
        for name, metric in metrics.items():
            for attempt in range(5):
                try:
                    v = asyncio.run(metric.single_turn_ascore(sample))
                    scores[name] = None if v is None or v != v else float(v)  # NaN → None
                    break
                except Exception as exc:  # noqa: BLE001 — free tier: rate limits, malformed judge output
                    wait = min(90, 8 * 2**attempt)
                    print(
                        f"  ragas {name} {rec.case_id}: {type(exc).__name__}: {str(exc)[:80]} — retry in {wait}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
            else:
                scores[name] = None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(scores), encoding="utf-8")
        out[rec.case_id] = scores
        print(f"  ragas {i}/{len(records)} {rec.case_id} {scores}", file=sys.stderr)
    return out


# ------------------------------------------------------------------ DeepEval


def _deepeval_scores(records: list[CaseRecord], judge_model: str) -> dict[str, dict[str, float | None]]:
    from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, FaithfulnessMetric
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase

    from safety_assistant.providers.llm import LLMMessage
    from safety_assistant.providers.llm.factory import get_llm_provider

    provider = get_llm_provider()
    assert provider is not None

    class GatewayLLM(DeepEvalBaseLLM):  # type: ignore[misc]
        def load_model(self):  # type: ignore[no-untyped-def]
            return provider

        def generate(self, prompt: str, schema: Any = None) -> Any:  # noqa: D102
            resp = provider.generate(
                [LLMMessage(role="user", content=prompt)], schema=schema, temperature=0.0, max_tokens=1500
            )
            return resp.parsed if schema is not None and resp.parsed is not None else resp.content

        async def a_generate(self, prompt: str, schema: Any = None) -> Any:  # noqa: D102
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:  # noqa: D102
            return judge_model

    judge = GatewayLLM()
    out: dict[str, dict[str, float | None]] = {}
    for i, rec in enumerate(records, 1):
        cache = _cache_key("deepeval", rec, judge_model)
        if cache.exists():
            out[rec.case_id] = json.loads(cache.read_text(encoding="utf-8"))
            continue
        tc = LLMTestCase(
            input=rec.query,
            actual_output=rec.answer or "",
            retrieval_context=rec.contexts[:8],
            expected_output=" ".join(rec.key_facts) or None,
        )
        scores: dict[str, float | None] = {}
        for name, metric in (
            ("faithfulness", FaithfulnessMetric(model=judge, async_mode=False, verbose_mode=False)),
            ("answer_relevancy", AnswerRelevancyMetric(model=judge, async_mode=False, verbose_mode=False)),
            ("contextual_precision", ContextualPrecisionMetric(model=judge, async_mode=False, verbose_mode=False)),
        ):
            for attempt in range(4):
                try:
                    metric.measure(tc)
                    scores[name] = float(metric.score) if metric.score is not None else None
                    break
                except Exception as exc:  # noqa: BLE001
                    wait = min(90, 8 * 2**attempt)
                    print(
                        f"  deepeval {name} {rec.case_id}: {type(exc).__name__}: {str(exc)[:80]} — retry in {wait}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
            else:
                scores[name] = None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(scores), encoding="utf-8")
        out[rec.case_id] = scores
        print(f"  deepeval {i}/{len(records)} {rec.case_id} {scores}", file=sys.stderr)
    return out


def _slice_table(
    records: list[CaseRecord], judged: dict[str, dict[str, float | None]], keys: list[str]
) -> dict[str, Any]:
    def agg(rs: list[CaseRecord]) -> dict[str, Any]:
        row: dict[str, Any] = {"n": sum(r.case_id in judged for r in rs)}
        for k in keys:
            row[k] = mean([judged[r.case_id].get(k) for r in rs if r.case_id in judged])
        return row

    return {
        "all": agg(records),
        **{s: agg([r for r in records if r.query_type == s]) for s in sorted({r.query_type for r in records})},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="evals/datasets/regulatory_v2.yaml")
    ap.add_argument("--slices", nargs="*", default=None, help="restrict to these query_type values")
    ap.add_argument("--source", choices=["human", "llm_generated", "llm_generated_reviewed"], default=None)
    ap.add_argument(
        "--limit", type=int, default=None, help="max cases (stable order) — answerable ones are sampled evenly"
    )
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--deepeval", action="store_true")
    ap.add_argument("--deepeval-limit", type=int, default=40)
    ap.add_argument("--ragas-limit", type=int, default=None, help="judge an even sample of N answerable records")
    ap.add_argument("--judge-model", default=None, help="gateway model for judging (default: LLM_MODEL)")
    ap.add_argument("--out", default="evals/results")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    dataset = load_dataset(pathlib.Path(args.dataset), source=args.source, query_types=args.slices or None)
    cases = list(dataset.cases)
    if args.limit and len(cases) > args.limit:
        step = len(cases) / args.limit
        cases = [cases[int(i * step)] for i in range(args.limit)]
    service = AnswerService()
    if service.llm is None:
        print("LLM_PROVIDER is not configured — judged evaluation needs the real generator", file=sys.stderr)
        return 2
    fp = pipeline_fingerprint(service, dataset)
    cache_dir = None if args.no_cache else pathlib.Path("evals/cache/answers")
    judge_model = args.judge_model or service.llm.model

    records: list[CaseRecord] = []
    t0 = time.perf_counter()
    with Session(get_engine(), expire_on_commit=False) as session:
        for i, case in enumerate(cases, 1):
            rec = run_case(session, service, case, cache_dir=cache_dir, fingerprint=fp)
            records.append(rec)
            if i % 5 == 0 or i == len(cases):
                print(f"  answered {i}/{len(cases)} ({(time.perf_counter() - t0) / 60:.1f} min)", file=sys.stderr)
    extra: dict[str, Any] = {"llm": f"{service.llm.name}:{service.llm.model}", "judge_model": judge_model}

    judgeable = [r for r in records if _judgeable(r)]
    if args.ragas_limit and len(judgeable) > args.ragas_limit:
        step = len(judgeable) / args.ragas_limit
        judgeable = [judgeable[int(i * step)] for i in range(args.ragas_limit)]
    if args.ragas:
        scores = _ragas_scores(judgeable, judge_model)
        extra["ragas"] = _slice_table(
            records, scores, ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        )
        for r in records:
            r.metrics.update({f"ragas_{k}": v for k, v in scores.get(r.case_id, {}).items()})
    if args.deepeval:
        subset = judgeable[: args.deepeval_limit]
        scores = _deepeval_scores(subset, judge_model)
        extra["deepeval"] = _slice_table(records, scores, ["faithfulness", "answer_relevancy", "contextual_precision"])
        for r in records:
            r.metrics.update({f"deepeval_{k}": v for k, v in scores.get(r.case_id, {}).items()})

    pricing_path = pathlib.Path("evals/pricing.yaml")
    if pricing_path.exists():
        import yaml

        extra["cost"] = cost_estimate(records, yaml.safe_load(pricing_path.read_text(encoding="utf-8")))
    rep = report(records, dataset=dataset, fingerprint=fp, extra=extra)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S+0000")
    path = out_dir / f"generation_{dataset.dataset_version}_{stamp}.json"
    path.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (out_dir / f"generation_{dataset.dataset_version}_latest.json").write_text(
        json.dumps(rep, indent=2, default=str), encoding="utf-8"
    )
    _print_summary(rep)
    print(f"written: {path}")
    return 0


def _print_summary(rep: dict[str, Any]) -> None:
    a = rep["aggregate"]
    print(f"{chr(10)}{rep['dataset_version']}  n={rep['n_cases']}  git {rep['git_sha']}")
    print(f"pipeline {rep['pipeline_fingerprint']}  llm {rep['llm']}  judge {rep.get('judge_model')}")
    print(f"modes {a['modes']}  p50 {a['latency_p50_ms']} ms")
    cols = [
        "refusal_correct",
        "citation_precision",
        "citation_hit",
        "fact_coverage",
        "evidence_coverage",
        "grounding_ok",
        "injection_resisted",
    ]
    hdr = f"{'slice':<32}{'n':>4}" + "".join(f"{c[:14]:>16}" for c in cols)
    print(hdr)

    def row(name: str, agg: dict[str, Any]) -> str:
        cells = "".join(f"{'-' if agg[c] is None else f'{agg[c]:.3f}':>16}" for c in cols)
        return f"{name:<32}{agg['n']:>4}{cells}"

    print(row("all", a))
    for s, agg in rep["by_slice"].items():
        print(row(s, agg))
    if rep.get("cost"):
        c = rep["cost"]
        blended = c["usd_per_query_blended"] or 0
        print(
            f"\ncost (reference prices as of {c['pricing_as_of']}): priced {c['priced_queries']} queries, "
            f"blended ${blended:.5f}/query; unpriced models: {c['unpriced_models']}"
        )
        for m, v in c["by_model"].items():
            print(
                f"  {m:<36} n={v['queries']:<4} ${v['usd_per_query']:.5f}/query  "
                f"${v['usd_per_1k_queries']:.2f}/1k  mean {v['latency_p_mean_ms']:.0f} ms"
            )
    for kind in ("ragas", "deepeval"):
        if kind in rep:
            print(f"\n{kind}:")
            for s, agg in rep[kind].items():
                cells = " ".join(f"{k}={'-' if v is None else f'{v:.3f}'}" for k, v in agg.items() if k != "n")
                print(f"  {s:<30} n={agg['n']:<4} {cells}")


if __name__ == "__main__":
    sys.exit(main())
