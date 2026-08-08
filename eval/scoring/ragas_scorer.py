"""RAGAS scoring for grounded Q&A categories (live SUT + cheap Portkey judge).

Categories scored here:
  factual_lookup, compliance_check, multi_hop, enumerative,
  cross_regulation, design_implication

Judge-model calls use the **pinned** eval judge
(``config/portkey/eval_judge_pinned.json`` via ``install_eval_judge_overflow``),
not production ``judge.json`` and not a multi-provider overflow ladder. Transient
failures retry via ``eval.eval_judge_retry`` (eval-infra only). Production
generation still uses FINAL_ANSWER / QUERY_REWRITE only; SUT answer calls are
unchanged. Every Portkey usage-log row is tagged
``call_kind`` = ``system_under_test`` | ``judge``.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Sequence

from eval.eval_judge_overflow import (
    eval_judge_overflow_primary_model,
    install_eval_judge_overflow,
)
from eval.gold import DEFAULT_GOLDEN, load_golden_set
from eval.scoring.custom_checks import (
    CUSTOM_HARD_GATE_CATEGORIES,
    run_custom_hard_gates,
)
from generation.answer import answer_question
from generation.llm_client import LLMClient, answering_fields_from_answer, llm_call_kind_scope
from retrieval.retrieve import RetrievedChunk, fetch_chunks_by_ids, retrieve

logger = logging.getLogger(__name__)

RAGAS_SCORE_CATEGORIES = frozenset(
    {
        "factual_lookup",
        "compliance_check",
        "multi_hop",
        "enumerative",
        "cross_regulation",
        "design_implication",
    }
)

# Default label for reports — pinned eval judge primary.
DEFAULT_RAGAS_JUDGE_MODEL = "gemini-2.5-flash"


class RagasScorerUnavailable(RuntimeError):
    """Raised when RAGAS / LangChain deps or judge routing cannot run."""


# Thousands groups like 1,000 or 1,000,000.5 — commas are separators, not decimals.
_THOUSANDS_GROUP_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?!\d)")


def _normalize_thousands_commas(text: str) -> str:
    """Strip thousands-separator commas; keep decimal points and other digits."""

    def _repl(match: re.Match[str]) -> str:
        return match.group(1).replace(",", "")

    return _THOUSANDS_GROUP_RE.sub(_repl, text or "")


def _has_substring(text: str, needle: str) -> bool:
    if not needle:
        return True
    if needle in text or needle.lower() in (text or "").lower():
        return True
    # "1000" ↔ "1,000" (regulation-style thousands separators).
    norm_text = _normalize_thousands_commas(text or "")
    norm_needle = _normalize_thousands_commas(needle)
    if norm_needle in norm_text:
        return True
    return norm_needle.lower() in norm_text.lower()


def substring_checks(
    answer: str,
    *,
    expected_answer_contains: Sequence[str] | None = None,
    must_not_contain: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Deterministic pass/fail against golden substring lists."""
    contains = [str(s) for s in (expected_answer_contains or []) if str(s).strip()]
    banned = [str(s) for s in (must_not_contain or []) if str(s).strip()]
    missing = [s for s in contains if not _has_substring(answer, s)]
    forbidden_hits = [s for s in banned if _has_substring(answer, s)]
    passed = not missing and not forbidden_hits
    return {
        "substring_pass": passed,
        "missing_expected": missing,
        "forbidden_hits": forbidden_hits,
        "expected_answer_contains": contains,
        "must_not_contain": banned,
    }


def _judge_model_name() -> str:
    """Report label for the overflow chain primary (env overrides for display only)."""
    return (
        (os.getenv("RAGAS_JUDGE_MODEL") or "").strip()
        or (os.getenv("EVAL_JUDGE_MODEL") or "").strip()
        or eval_judge_overflow_primary_model()
        or DEFAULT_RAGAS_JUDGE_MODEL
    )


def _answer_model_basename(model: str) -> str:
    return (model or "").strip().split("/")[-1].lower()


def _warn_if_judge_matches_answer(answer_model: str, judge_model: str) -> None:
    a = _answer_model_basename(answer_model)
    j = _answer_model_basename(judge_model)
    if a and j and a == j:
        logger.warning(
            "RAGAS judge model %r matches answer model %r — self-preference risk; "
            "set RAGAS_JUDGE_MODEL to a cheaper / different class "
            "(e.g. gemini-2.5-flash or a small NIM instruct model).",
            judge_model,
            answer_model,
        )


def _portkey_judge_langchain(client: LLMClient):
    """LangChain chat model → pinned eval judge + usage log."""
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from pydantic import ConfigDict, Field
    except ImportError as exc:
        raise RagasScorerUnavailable(
            "langchain-core required for RAGAS judge wrapper — uv sync --extra eval"
        ) from exc

    class PortkeyJudgeChatModel(BaseChatModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        llm_client: Any = Field(exclude=True)
        bound_model: str = DEFAULT_RAGAS_JUDGE_MODEL

        @property
        def _llm_type(self) -> str:
            return "portkey-eval-judge-overflow"

        @property
        def _identifying_params(self) -> dict[str, Any]:
            return {"model": self.bound_model, "role": "judge", "config": "eval_judge_overflow"}

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            converted: list[dict[str, str]] = []
            for msg in messages:
                mtype = getattr(msg, "type", "") or ""
                if mtype == "system":
                    role = "system"
                elif mtype in {"ai", "assistant"}:
                    role = "assistant"
                else:
                    role = "user"
                converted.append({"role": role, "content": str(msg.content or "")})
            with llm_call_kind_scope("judge"):
                from eval.eval_judge_retry import judge_with_eval_retry

                result = judge_with_eval_retry(
                    self.llm_client, messages=converted, question=""
                )
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=result.text or ""))]
            )

    return PortkeyJudgeChatModel(llm_client=client, bound_model=_judge_model_name())


def _ragas_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        except ImportError as exc:
            raise RagasScorerUnavailable(
                "Install eval extras for embeddings: uv sync --extra eval"
            ) from exc
    emb_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    return HuggingFaceEmbeddings(model_name=emb_model)


def _build_ragas_wrappers(client: LLMClient):
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RagasScorerUnavailable(
            "Install eval extras: uv sync --extra eval (ragas, langchain-*)"
        ) from exc
    install_eval_judge_overflow(client)
    llm = LangchainLLMWrapper(_portkey_judge_langchain(client))
    embeddings = LangchainEmbeddingsWrapper(_ragas_embeddings())
    return llm, embeddings


def _normalize_explicit_ref_chunks(
    raw: Any,
) -> list[dict[str, str]]:
    """Parse golden-set ``ground_truth_reference_chunks`` (strings or dicts)."""
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append({"chunk_id": "", "text": text})
            continue
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("chunk") or item.get("content") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "chunk_id": str(item.get("chunk_id") or "").strip(),
                    "text": text,
                }
            )
    return out


def _ground_truth_reference(case: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """Build RAGAS ground_truth for context_recall / context_precision.

    Preference order:
    1. Explicit ``ground_truth_reference_chunks`` on the golden case (verbatim
       regulation excerpts — preferred so empty ``expected_chunk_ids`` do not
       collapse to weak substring hints).
    2. Texts fetched for ``expected_chunk_ids``.
    3. Weak fallbacks from ``expected_answer_contains`` / ``expected_behavior``.
    """
    explicit = _normalize_explicit_ref_chunks(case.get("ground_truth_reference_chunks"))
    if explicit:
        joined = "\n\n".join(
            (
                f"[{c['chunk_id']}]\n{c['text']}".strip()
                if c.get("chunk_id")
                else c["text"]
            )
            for c in explicit
            if c.get("text")
        ).strip()
        if joined:
            return joined, explicit

    expected_ids = [str(x).strip() for x in (case.get("expected_chunk_ids") or []) if str(x).strip()]
    ref_chunks: list[dict[str, str]] = []
    if expected_ids:
        try:
            fetched = fetch_chunks_by_ids(expected_ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_chunks_by_ids failed for %s: %s", case.get("id"), exc)
            fetched = []
        by_id = {c.chunk_id: c for c in fetched if c.chunk_id}
        for cid in expected_ids:
            chunk = by_id.get(cid)
            text = (chunk.text if chunk else "") or ""
            ref_chunks.append({"chunk_id": cid, "text": text})
        joined = "\n\n".join(
            f"[{c['chunk_id']}]\n{c['text']}".strip() for c in ref_chunks if c["text"]
        ).strip()
        if joined:
            return joined, ref_chunks
        # IDs known but texts missing — still pass ids as a weak reference string.
        return "Expected evidence chunk_ids: " + ", ".join(expected_ids), ref_chunks

    contains = [str(s) for s in (case.get("expected_answer_contains") or []) if str(s).strip()]
    if contains:
        return "Reference answer must include: " + "; ".join(contains), ref_chunks
    beh = str(case.get("expected_behavior") or case.get("answer") or case.get("ground_truth") or "").strip()
    return beh or "(no ground-truth reference)", ref_chunks


def _run_live_sut(
    case: dict[str, Any],
    *,
    llm: LLMClient,
) -> dict[str, Any]:
    """Retrieve + answer; tag all Portkey rows as system_under_test."""
    question = str(case.get("question") or "").strip()
    reg = case.get("regulation_scope") or case.get("regulation_id")
    with llm_call_kind_scope("system_under_test"):
        chunks: list[RetrievedChunk] = retrieve(
            question,
            regulation_id=reg,
            rewrite=True,
            do_rerank=True,
            small_to_big=True,
        )
        ans = answer_question(
            question,
            regulation_id=reg,
            llm=llm,
            chunks=chunks,
            persist_turn=False,
            skip_answer_cache=True,
        )
    retrieved = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text or "",
            "regulation_id": c.regulation_id or "",
            "section_number": c.section_number or "",
            "score": float(c.score or 0.0),
        }
        for c in chunks
    ]
    cited = [
        {
            "chunk_id": s.chunk_id,
            "text": s.text or "",
            "citation": s.citation or "",
            "regulation_id": s.regulation_id or "",
            "section_number": s.section_number or "",
        }
        for s in (ans.sources or [])
    ]
    return {
        "question": question,
        "answer": ans.answer or "",
        "retrieved_chunks": retrieved,
        "cited_chunk_ids": [s["chunk_id"] for s in cited if s.get("chunk_id")],
        "cited_chunks": cited,
        "contexts": [c["text"] for c in retrieved if (c.get("text") or "").strip()]
        or [s["text"] for s in cited if (s.get("text") or "").strip()]
        or ["(no retrieved context)"],
        "not_found": bool(ans.not_found),
        "failure_kind": ans.failure_kind,
        **answering_fields_from_answer(ans),
    }


def _score_ragas_single(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    llm_wrapper: Any,
    embeddings_wrapper: Any,
) -> dict[str, float | None]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RagasScorerUnavailable(
            "Install eval extras: uv sync --extra eval (ragas, datasets)"
        ) from exc

    ds = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
    )
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_wrapper,
        embeddings=embeddings_wrapper,
    )
    scores: dict[str, float | None] = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }
    try:
        df = result.to_pandas()
        row = df.to_dict(orient="records")[0] if len(df) else {}
    except Exception:  # noqa: BLE001
        row = result if isinstance(result, dict) else {}
    for key in scores:
        raw = row.get(key) if isinstance(row, dict) else getattr(result, key, None)
        if raw is None:
            continue
        try:
            scores[key] = round(float(raw), 4)
        except (TypeError, ValueError):
            scores[key] = None

    # Surface NaN faithfulness with the answer text — useful for diagnosing whether
    # claim-extraction failed on short/decline answers vs long structured ones.
    faith = scores.get("faithfulness")
    if faith is not None and isinstance(faith, float) and math.isnan(faith):
        preview = (answer or "").replace("\n", " ").strip()
        if len(preview) > 400:
            preview = preview[:400] + "…"
        logger.warning(
            "RAGAS faithfulness=NaN answer_len=%d preview=%r",
            len(answer or ""),
            preview,
        )
    return scores


def _chunk_id_overlap_metrics(
    *,
    retrieved_ids: Sequence[str],
    expected_ids: Sequence[str],
    cited_ids: Sequence[str],
) -> dict[str, float | None]:
    """Deterministic id-level precision/recall vs expected_chunk_ids (diagnostic)."""
    expected = {str(x).strip() for x in expected_ids if str(x).strip()}
    if not expected:
        return {
            "context_precision_vs_expected_ids": None,
            "context_recall_vs_expected_ids": None,
            "citation_recall_vs_expected_ids": None,
        }
    retrieved = {str(x).strip() for x in retrieved_ids if str(x).strip()}
    cited = {str(x).strip() for x in cited_ids if str(x).strip()}
    def _pr(hit: set[str], denom: set[str]) -> float | None:
        if not denom:
            return None
        return round(len(hit & expected) / len(denom), 4)

    return {
        "context_precision_vs_expected_ids": _pr(retrieved, retrieved),
        "context_recall_vs_expected_ids": round(len(retrieved & expected) / len(expected), 4),
        "citation_recall_vs_expected_ids": round(len(cited & expected) / len(expected), 4)
        if cited
        else 0.0,
    }


def score_case(
    case: dict[str, Any],
    *,
    llm: LLMClient | None = None,
    skip_ragas: bool = False,
) -> dict[str, Any]:
    """Run one golden case through the live system and score with RAGAS + substrings."""
    category = str(case.get("category") or "").strip().lower()
    if category and category not in RAGAS_SCORE_CATEGORIES:
        raise ValueError(
            f"score_case: category {category!r} not in RAGAS_SCORE_CATEGORIES"
        )

    client = llm or LLMClient()
    judge_model = _judge_model_name()
    # Ensure judge config picks up RAGAS_JUDGE_MODEL even when .env was loaded later.
    os.environ.setdefault("RAGAS_JUDGE_MODEL", judge_model)

    live = _run_live_sut(case, llm=client)
    _warn_if_judge_matches_answer(live.get("sut_model") or "", judge_model)

    ground_truth, ref_chunks = _ground_truth_reference(case)
    substr = substring_checks(
        live["answer"],
        expected_answer_contains=case.get("expected_answer_contains"),
        must_not_contain=case.get("must_not_contain"),
    )
    id_metrics = _chunk_id_overlap_metrics(
        retrieved_ids=[c["chunk_id"] for c in live["retrieved_chunks"]],
        expected_ids=case.get("expected_chunk_ids") or [],
        cited_ids=live["cited_chunk_ids"],
    )

    hard_gates = None
    if str(case.get("category") or "").strip().lower() in CUSTOM_HARD_GATE_CATEGORIES:
        hard_gates = run_custom_hard_gates(
            case,
            answer=live["answer"],
            retrieved_chunk_ids=[c["chunk_id"] for c in live["retrieved_chunks"]],
            cited_chunk_ids=live["cited_chunk_ids"],
            cited_sources=live["cited_chunks"],
        )

    ragas_scores: dict[str, float | None] = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }
    ragas_error: str | None = None
    if not skip_ragas and (os.getenv("RAGAS_SKIP") or "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        try:
            llm_w, emb_w = _build_ragas_wrappers(client)
            ragas_scores = _score_ragas_single(
                question=live["question"],
                answer=live["answer"],
                contexts=live["contexts"],
                ground_truth=ground_truth,
                llm_wrapper=llm_w,
                embeddings_wrapper=emb_w,
            )
        except Exception as exc:  # noqa: BLE001
            ragas_error = str(exc)
            logger.exception("RAGAS scoring failed for %s", case.get("id"))

    overall_pass = bool(substr["substring_pass"])
    if hard_gates is not None and hard_gates.get("applicable"):
        overall_pass = overall_pass and bool(hard_gates.get("pass"))

    return {
        "id": case.get("id"),
        "category": category or case.get("category"),
        "severity": case.get("severity"),
        "question": live["question"],
        "answer": live["answer"],
        "retrieved_chunks": live["retrieved_chunks"],
        "cited_chunk_ids": live["cited_chunk_ids"],
        "cited_chunks": live["cited_chunks"],
        "expected_chunk_ids": list(case.get("expected_chunk_ids") or []),
        "ground_truth_reference_chunks": ref_chunks,
        "answering_model": live.get("answering_model") or "",
        "answering_provider_was_fallback": bool(
            live.get("answering_provider_was_fallback") or False
        ),
        "sut_model": live.get("sut_model") or "",
        "sut_provider": live.get("sut_provider") or "",
        "judge_model": judge_model,
        "not_found": live["not_found"],
        "failure_kind": live["failure_kind"],
        "ragas": ragas_scores,
        "chunk_id_metrics": id_metrics,
        "substring_checks": substr,
        "custom_hard_gates": hard_gates,
        "pass": overall_pass,
        "ragas_error": ragas_error,
    }


def score_cases(
    cases: Sequence[dict[str, Any]] | None = None,
    *,
    gold_path: Any = None,
    llm: LLMClient | None = None,
    skip_ragas: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Score all (or filtered) RAGAS categories from the golden set."""
    if cases is None:
        cases = load_golden_set(gold_path or DEFAULT_GOLDEN)
    selected = [
        c
        for c in cases
        if str(c.get("category") or "").strip().lower() in RAGAS_SCORE_CATEGORIES
    ]
    if limit is not None:
        selected = selected[: max(0, int(limit))]

    client = llm or LLMClient()
    per_case: list[dict[str, Any]] = []
    for case in selected:
        logger.info("RAGAS score_case %s (%s)", case.get("id"), case.get("category"))
        per_case.append(score_case(case, llm=client, skip_ragas=skip_ragas))

    averages: dict[str, float] = {}
    for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        vals = [
            float(c["ragas"][key])
            for c in per_case
            if c.get("ragas") and c["ragas"].get(key) is not None
        ]
        if vals:
            averages[key] = round(sum(vals) / len(vals), 4)

    n_pass = sum(1 for c in per_case if c.get("pass"))
    return {
        "framework": "ragas",
        "judge_model": _judge_model_name(),
        "categories": sorted(RAGAS_SCORE_CATEGORIES),
        "n_cases": len(per_case),
        "n_pass": n_pass,
        "substring_pass_rate": round(n_pass / len(per_case), 4) if per_case else None,
        "averages": averages,
        "cases": per_case,
    }


# Back-compat alias used by eval.scoring.ragas_scores
run_ragas_scorer = score_cases
