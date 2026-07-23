"""RAGAS scoring with Groq-only judge + answerable/abstention tracks."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)

ANSWERABLE = "answerable"
ABSTENTION = "abstention"

# Binary refusal cues — keep simple; do not weaken system abstention behavior.
_REFUSAL_RE = re.compile(
    r"(?i)("
    r"false premise|do(?:es)? not (contain|apply|impose)|cannot (be verified|confirm|find)|"
    r"outside (the )?(indexed|unece)|out of scope|not (present|available|in the)|"
    r"will not answer|cannot answer|not frontal|lateral/?side|"
    r"no(,| —| -)? the retrieved|I cannot|not in (the )?(retrieved|corpus|provided)"
    r")"
)

# Affirming a known hallucination trap is an incorrect refusal.
_CONFIRM_TRAP_RE = re.compile(
    r"(?i)(yes[,.]?\s*(it|the)|confirm(s|ed)?|sets a maximum|550\s*hic.*(is|are)\s*(correct|true|specified))"
)


class RagasScoringError(RuntimeError):
    """RAGAS judge unavailable or returned incomplete scores."""


def _strip_answer(answer: str) -> str:
    from core.generation.answer_formatter import strip_scoring_artifacts

    return strip_scoring_artifacts(answer)


def _trim_contexts(
    contexts: list[str],
    *,
    max_chars_each: int = 600,
    max_total_chars: int = 2800,
) -> list[str]:
    """Cap context size so judge prompts fit Groq free-tier TPM."""
    trimmed: list[str] = []
    total = 0
    for raw in contexts:
        text = (raw or "").strip()
        if not text:
            continue
        if len(text) > max_chars_each:
            text = text[: max_chars_each - 1] + "…"
        if total + len(text) > max_total_chars:
            remain = max_total_chars - total
            if remain < 80:
                break
            text = text[: remain - 1] + "…"
            trimmed.append(text)
            break
        trimmed.append(text)
        total += len(text)
    return trimmed or [""]


class _GroqNClamp:
    """Force n=1 — Groq rejects n>1; RAGAS may request more via bind()."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def bind(self, **kwargs: Any) -> Any:
        out = dict(kwargs)
        out["n"] = 1
        out.pop("model_kwargs", None)
        return _GroqNClamp(self._inner.bind(**out))

    def _clean(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        out = dict(kwargs)
        out.pop("model_kwargs", None)
        return out

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.invoke(*args, **self._clean(kwargs))

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.ainvoke(*args, **self._clean(kwargs))

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.generate(*args, **self._clean(kwargs))

    async def agenerate(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.agenerate(*args, **self._clean(kwargs))

    def generate_prompt(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.generate_prompt(*args, **self._clean(kwargs))

    async def agenerate_prompt(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.agenerate_prompt(*args, **self._clean(kwargs))


def _resolve_embed_model(model: str) -> str:
    name = "models--" + model.replace("/", "--")
    roots = []
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]))
    roots.append(Path.home() / ".cache" / "huggingface")
    roots.append(Path(__file__).resolve().parents[1] / "docker" / "hf_cache")
    for root in roots:
        for snap_root in (root / "hub" / name / "snapshots", root / name / "snapshots"):
            if not snap_root.is_dir():
                continue
            snaps = [p for p in snap_root.iterdir() if p.is_dir()]
            if snaps:
                for snap in snaps:
                    if (snap / "config.json").is_file():
                        return str(snap)
                return str(snaps[0])
    return model


def _build_judge() -> tuple[Any, Any]:
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RagasScoringError("GROQ_API_KEY is required for RAGAS (Groq-only).")

    try:
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RagasScoringError(f"RAGAS dependencies missing: {exc}") from exc

    judge_model = (
        os.getenv("RAGAS_JUDGE_MODEL")
        or os.getenv("GROQ_MODEL_INSTANT")
        or "llama-3.1-8b-instant"
    ).strip()
    max_tokens = int(os.getenv("RAGAS_JUDGE_MAX_TOKENS", "2048"))
    embed_name = _resolve_embed_model(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    chat = ChatGroq(
        model=judge_model,
        api_key=api_key,
        temperature=0,
        max_tokens=max_tokens,
        n=1,
    )
    llm = LangchainLLMWrapper(_GroqNClamp(chat))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embed_name))
    logger.info("RAGAS judge=Groq model={} embeddings={}", judge_model, embed_name)
    return llm, embeddings


def score_abstention(answer: str, *, case_id: str = "") -> dict[str, Any]:
    """Binary correct-refusal for abstention-track questions (no RAGAS)."""
    text = _strip_answer(answer or "")
    refused = bool(_REFUSAL_RE.search(text))
    # Hallucination traps fail if the model affirms the invented claim.
    if case_id.startswith("q07") and _CONFIRM_TRAP_RE.search(text) and "not" not in text.lower()[:80]:
        refused = False
    if case_id.startswith("q08") and re.search(r"(?i)\b(hic|chest|pole)\b.+\d+", text):
        # Inventing FMVSS numbers without refusal → incorrect
        if not refused:
            refused = False
    return {
        "correct_refusal": refused,
        "refusal_score": 1.0 if refused else 0.0,
        "final_score": 1.0 if refused else 0.0,
    }


def run_ragas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score answerable cases with RAGAS; abstention with binary refusal."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    for rec in records:
        track = rec.get("track") or ANSWERABLE
        rec["track"] = track
        if track == ABSTENTION:
            rec["metrics"] = score_abstention(rec.get("answer", ""), case_id=str(rec.get("id", "")))
            rec["metrics_source"] = "abstention_binary"
            continue

    answerable = [r for r in records if r["track"] == ANSWERABLE]
    if not answerable:
        return records

    llm, embeddings = _build_judge()
    ds = Dataset.from_dict(
        {
            "question": [r["query"] for r in answerable],
            "answer": [_strip_answer(r.get("answer", "")) for r in answerable],
            "contexts": [
                _trim_contexts(
                    [
                        (c.get("snippet") or "")
                        for c in (r.get("retrieved_chunks") or [])
                        if (c.get("snippet") or "").strip()
                    ]
                    or [r.get("retrieved_context") or ""]
                )
                for r in answerable
            ],
            "ground_truth": [r.get("ground_truth", "") for r in answerable],
        }
    )

    try:
        result = evaluate(
            ds,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            ],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=1, max_wait=180, max_retries=6),
        )
        df = result.to_pandas()
    except Exception as exc:
        raise RagasScoringError(f"RAGAS evaluate() failed: {exc}") from exc

    # Retry individual rows that returned NaN (transient Groq/RAGAS failures).
    for attempt in range(2):
        missing_rows: list[int] = []
        for idx in range(len(answerable)):
            for key in METRICS:
                if key not in df.columns or float(df.iloc[idx][key]) != float(df.iloc[idx][key]):
                    missing_rows.append(idx)
                    break
        if not missing_rows:
            break
        logger.warning(
            "RAGAS NaN/missing on rows {} — retry {}/2",
            missing_rows,
            attempt + 1,
        )
        sub = [answerable[i] for i in missing_rows]
        sub_ds = Dataset.from_dict(
            {
                "question": [r["query"] for r in sub],
                "answer": [_strip_answer(r.get("answer", "")) for r in sub],
                "contexts": [
                    _trim_contexts(
                        [
                            (c.get("snippet") or "")
                            for c in (r.get("retrieved_chunks") or [])
                            if (c.get("snippet") or "").strip()
                        ]
                        or [r.get("retrieved_context") or ""]
                    )
                    for r in sub
                ],
                "ground_truth": [r.get("ground_truth", "") for r in sub],
            }
        )
        try:
            sub_result = evaluate(
                sub_ds,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    answer_correctness,
                ],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(max_workers=1, max_wait=180, max_retries=6),
            )
            sub_df = sub_result.to_pandas()
        except Exception as exc:
            raise RagasScoringError(f"RAGAS retry failed: {exc}") from exc
        for j, idx in enumerate(missing_rows):
            for key in METRICS:
                if key in sub_df.columns:
                    df.iat[idx, df.columns.get_loc(key)] = sub_df.iloc[j][key]

    for idx, rec in enumerate(answerable):
        scores: dict[str, float] = {}
        missing: list[str] = []
        for key in METRICS:
            if key not in df.columns:
                missing.append(key)
                continue
            val = float(df.iloc[idx][key])
            if val != val:  # NaN
                missing.append(key)
                continue
            scores[key] = round(val, 4)
        if missing:
            raise RagasScoringError(
                f"Incomplete RAGAS scores for {rec.get('id')}: missing/NaN={missing}"
            )
        scores["final_score"] = round(sum(scores[k] for k in METRICS) / len(METRICS), 4)
        rec["metrics"] = scores
        rec["metrics_source"] = "ragas"
    return records


def overall_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Legacy pooled summary (answerable RAGAS only when tracks present)."""
    answerable = [r for r in records if r.get("track") == ANSWERABLE]
    abstention = [r for r in records if r.get("track") == ABSTENTION]
    scored = answerable or records

    out: dict[str, Any] = {}
    if answerable or all("faithfulness" in (r.get("metrics") or {}) for r in scored):
        for key in METRICS:
            vals = [r["metrics"][key] for r in scored if key in r.get("metrics", {})]
            if vals:
                out[f"avg_{key}"] = round(sum(vals) / len(vals), 4)
        finals = [r["metrics"]["final_score"] for r in scored]
        out["avg_final_score"] = round(sum(finals) / len(finals), 4)
        out["pass_count"] = sum(1 for s in finals if s >= 0.65)
        out["fail_count"] = len(finals) - out["pass_count"]
    out["judge"] = "groq"
    out["framework"] = "ragas"
    out["n_questions"] = len(records)
    out["n_answerable"] = len(answerable)
    out["n_abstention"] = len(abstention)
    return out


def track_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate answerable (RAGAS) and abstention (binary refusal) tracks."""
    answerable = [r for r in records if r.get("track") == ANSWERABLE]
    abstention = [r for r in records if r.get("track") == ABSTENTION]

    ans: dict[str, Any] = {"n_questions": len(answerable), "framework": "ragas", "judge": "groq"}
    if answerable:
        for key in METRICS:
            vals = [r["metrics"][key] for r in answerable]
            ans[f"avg_{key}"] = round(sum(vals) / len(vals), 4)
        finals = [r["metrics"]["final_score"] for r in answerable]
        ans["avg_final_score"] = round(sum(finals) / len(finals), 4)
        ans["pass_count"] = sum(1 for s in finals if s >= 0.65)
        ans["fail_count"] = len(finals) - ans["pass_count"]

    correct = sum(1 for r in abstention if r["metrics"].get("correct_refusal"))
    abs_out: dict[str, Any] = {
        "n_questions": len(abstention),
        "framework": "abstention_binary",
        "correct_refusals": correct,
        "incorrect_refusals": len(abstention) - correct,
        "accuracy": round(correct / len(abstention), 4) if abstention else 0.0,
    }
    return {"answerable": ans, "abstention": abs_out}
