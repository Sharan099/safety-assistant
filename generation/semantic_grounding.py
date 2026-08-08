"""Semantic groundedness: claim-vs-chunk support (layer 2 after chunk_id checks).

Fix 4 ensures citation_chunk_id ∈ retrieved set. This module checks whether the
*claim text* is actually substantiated by that chunk — especially for confident
negative / absence assertions.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_DISAGREEMENTS_PATH = ROOT / "eval" / "results" / "semantic_groundedness_disagreements.jsonl"

# Confident absence / negative relationship claims that need explicit support.
_NEGATIVE_CLAIM_RE = re.compile(
    r"(?ix)"
    r"("
    r"\bno\s+direct\s+(?:relationship|link|connection|bearing)\b"
    r"|\bno\s+(?:relationship|link|connection)\b"
    r"|\b(?:are|is)\s+unrelated\b"
    r"|\bnot\s+(?:directly\s+)?related\b"
    r"|\bthere\s+is\s+no\s+(?:direct\s+)?(?:relationship|link|connection|requirement|provision)\b"
    r"|\bno\s+requirement\s+exists\b"
    r"|\b(?:is|are)\s+not\s+(?:covered|addressed|mentioned|specified)\b"
    r"|\bnot\s+covered\s+(?:by|in|under|within)\b"
    r"|\bdoes\s+not\s+(?:cover|address|relate(?:\s+to)?|mention|apply\s+to)\b"
    r"|\bthis\s+is\s+not\s+covered\b"
    r")"
)

# Chunk must *state* absence / non-applicability — silence is not enough.
_EXPLICIT_ABSENCE_RE = re.compile(
    r"(?ix)"
    r"("
    r"\bshall\s+not\s+apply\b"
    r"|\bdoes\s+not\s+apply\b"
    r"|\bnot\s+applicable\b"
    r"|\bout\s+of\s+scope\b"
    r"|\bexcluded\s+from\b"
    r"|\bthis\s+regulation\s+does\s+not\b"
    r"|\bnot\s+within\s+the\s+scope\b"
    r"|\bno\s+requirement\s+(?:is|shall\s+be)\b"
    r")"
)

_SUPPORT_JUDGE_SYSTEM = """\
You are a strict groundedness judge for UNECE regulation Q&A.
Reply with exactly one word: YES or NO.

YES only if the PASSAGE explicitly supports the CLAIM (same fact, limit, or
stated absence). Silence, topic overlap, or a loosely related approval mark /
annex table is NOT support.
If the CLAIM is a negative/absence statement ("no relationship", "not covered",
"no requirement"), answer YES only when the PASSAGE itself states that absence.
"""


@dataclass
class SegmentSupportResult:
    claim: str
    chunk_id: str
    supported: bool
    method: str  # "heuristic" | "llm" | "skip"
    reason: str = ""
    llm_raw: str = ""


@dataclass
class SemanticGroundednessReport:
    question: str
    results: list[SegmentSupportResult] = field(default_factory=list)
    dropped_claims: list[str] = field(default_factory=list)
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_supported(self) -> bool:
        return all(r.supported for r in self.results) if self.results else True


def is_negative_or_absence_claim(text: str) -> bool:
    return bool(_NEGATIVE_CLAIM_RE.search(text or ""))


def chunk_explicitly_states_absence(chunk_text: str) -> bool:
    return bool(_EXPLICIT_ABSENCE_RE.search(chunk_text or ""))


def heuristic_claim_supported(claim: str, chunk_text: str) -> tuple[bool, str]:
    """Fast local check — focuses on the high-risk negative-claim failure mode.

    Positive claims are treated as provisionally supported here (LLM/eval layer
    does deeper sampling). Negative claims require explicit absence language in
    the chunk.
    """
    claim = (claim or "").strip()
    chunk_text = chunk_text or ""
    if not claim:
        return False, "empty_claim"
    if is_negative_or_absence_claim(claim):
        if chunk_explicitly_states_absence(chunk_text):
            return True, "negative_claim_with_explicit_absence_in_chunk"
        return False, "negative_claim_without_explicit_absence_in_chunk"
    return True, "non_negative_provisional"


def filter_unsupported_negative_segments(
    segments: Sequence[Any],
    chunks_by_id: dict[str, Any],
) -> tuple[list[Any], list[str]]:
    """Drop answer segments that assert absence without chunk support.

    ``segments`` items need ``.text`` and ``.citation_chunk_id`` (AnswerSegment).
    """
    kept: list[Any] = []
    dropped: list[str] = []
    for seg in segments:
        claim = (getattr(seg, "text", None) or "").strip()
        cid = (getattr(seg, "citation_chunk_id", None) or "").strip()
        chunk = chunks_by_id.get(cid)
        chunk_text = ""
        if chunk is not None:
            chunk_text = getattr(chunk, "text", None) or getattr(chunk, "enriched_text", None) or ""
        ok, reason = heuristic_claim_supported(claim, str(chunk_text))
        if not ok:
            dropped.append(claim)
            logger.warning(
                "dropped unsupported negative/absence claim (chunk_id=%s reason=%s): %r",
                cid,
                reason,
                claim[:160],
            )
            continue
        kept.append(seg)
    return kept, dropped


def drop_unsupported_audit_segments(
    segments: Sequence[Any],
    report: SemanticGroundednessReport,
) -> tuple[list[Any], list[str]]:
    """Fail-closed: drop segments the semantic audit marked unsupported.

    When the LLM judge is uncertain/unparseable it reports ``supported=False``;
    those claims are declined rather than kept.
    """
    unsupported_ids: set[str] = set()
    unsupported_claims: set[str] = set()
    for res in report.results:
        if res.supported:
            continue
        if res.chunk_id:
            unsupported_ids.add(res.chunk_id)
        claim = (res.claim or "").strip()
        if claim:
            unsupported_claims.add(claim)
    if not unsupported_ids and not unsupported_claims:
        return list(segments), []

    kept: list[Any] = []
    dropped: list[str] = []
    for seg in segments:
        claim = (getattr(seg, "text", None) or "").strip()
        cid = (getattr(seg, "citation_chunk_id", None) or "").strip()
        if (cid and cid in unsupported_ids) or (claim and claim in unsupported_claims):
            dropped.append(claim or cid or "(empty)")
            continue
        kept.append(seg)
    return kept, dropped


def _parse_yes_no(raw: str) -> bool | None:
    t = (raw or "").strip().upper()
    t = re.sub(r"^[^A-Z]*", "", t)
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    return None


def llm_claim_supported(
    claim: str,
    chunk_text: str,
    *,
    llm: Any,
    question: str = "",
) -> SegmentSupportResult:
    """Cheap small-model yes/no: does this chunk support this claim?"""
    from generation.llm_client import LLMRole

    passage = " ".join((chunk_text or "").split())[:2500]
    user = (
        f"QUESTION (context only): {question or '(n/a)'}\n\n"
        f"CLAIM:\n{claim}\n\n"
        f"PASSAGE:\n{passage}\n\n"
        "Does the PASSAGE explicitly support the CLAIM? Reply YES or NO:"
    )
    try:
        result = llm.complete(
            messages=[
                {"role": "system", "content": _SUPPORT_JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            role=LLMRole.REWRITE,  # small / cheap model
            question=f"groundedness:{claim[:80]}",
            chunk_ids=[],
            max_tokens=8,
            temperature=0.0,
            seed=42,
            skip_cache=True,
        )
        raw = (result.text or "").strip()
        parsed = _parse_yes_no(raw)
        if parsed is None:
            # Fail-closed: uncertain/ambiguous judge output → treat as unsupported
            # (decline rather than fabricate / keep ungrounded claims).
            return SegmentSupportResult(
                claim=claim,
                chunk_id="",
                supported=False,
                method="llm",
                reason="unparseable_judge_response_fail_closed",
                llm_raw=raw,
            )
        return SegmentSupportResult(
            claim=claim,
            chunk_id="",
            supported=parsed,
            method="llm",
            reason="llm_yes" if parsed else "llm_no",
            llm_raw=raw,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic groundedness LLM check failed: %s", exc)
        return SegmentSupportResult(
            claim=claim,
            chunk_id="",
            supported=False,  # fail-closed on judge error
            method="skip",
            reason=f"llm_error_fail_closed:{exc}",
        )


def audit_segments_semantic(
    *,
    question: str,
    segments: Sequence[Any],
    chunks_by_id: dict[str, Any],
    llm: Any | None = None,
    use_llm: bool | None = None,
) -> SemanticGroundednessReport:
    """Heuristic on all segments; optional LLM yes/no when enabled."""
    if use_llm is None:
        use_llm = (os.getenv("SEMANTIC_GROUNDEDNESS_LLM") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    report = SemanticGroundednessReport(question=question)
    for seg in segments:
        claim = (getattr(seg, "text", None) or "").strip()
        cid = (getattr(seg, "citation_chunk_id", None) or "").strip()
        chunk = chunks_by_id.get(cid)
        chunk_text = ""
        if chunk is not None:
            chunk_text = getattr(chunk, "text", None) or getattr(chunk, "enriched_text", None) or ""
        ok, reason = heuristic_claim_supported(claim, str(chunk_text))
        h_res = SegmentSupportResult(
            claim=claim,
            chunk_id=cid,
            supported=ok,
            method="heuristic",
            reason=reason,
        )
        report.results.append(h_res)
        if not ok:
            report.dropped_claims.append(claim)

        if use_llm and llm is not None and claim:
            llm_res = llm_claim_supported(claim, str(chunk_text), llm=llm, question=question)
            llm_res.chunk_id = cid
            # Fail-closed: LLM NO / uncertain overrides provisional heuristic keep.
            if not llm_res.supported:
                h_res.supported = False
                h_res.reason = f"llm_override:{llm_res.reason}"
                if claim and claim not in report.dropped_claims:
                    report.dropped_claims.append(claim)
            if llm_res.supported != ok and llm_res.method == "llm":
                disagreement = {
                    "question": question,
                    "claim": claim,
                    "chunk_id": cid,
                    "heuristic_supported": ok,
                    "heuristic_reason": reason,
                    "llm_supported": llm_res.supported,
                    "llm_raw": llm_res.llm_raw,
                    "chunk_preview": " ".join(str(chunk_text).split())[:240],
                }
                report.disagreements.append(disagreement)
                logger.warning(
                    "semantic groundedness disagreement heuristic=%s llm=%s claim=%r",
                    ok,
                    llm_res.supported,
                    claim[:120],
                )
    return report


def log_semantic_disagreements(
    disagreements: Sequence[dict[str, Any]],
    *,
    path: Path | None = None,
) -> None:
    if not disagreements:
        return
    out = path or SEMANTIC_DISAGREEMENTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for rec in disagreements:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("logged %d semantic groundedness disagreement(s) → %s", len(disagreements), out)


def run_eval_semantic_sample(
    records: Sequence[dict[str, Any]],
    *,
    llm: Any,
    sample_n: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Sample generation-eval answers and run cheap claim-support checks.

    Expects records with ``question``, ``answer``, and ``sources`` (list of
    dicts with chunk_id + text) — as produced by ``collect_generation_records``.
    """
    import random

    from generation.answer import AnswerSegment

    n = sample_n
    if n is None:
        try:
            n = int((os.getenv("SEMANTIC_GROUNDEDNESS_SAMPLE") or "5").strip() or "5")
        except ValueError:
            n = 5
    n = max(0, n)
    if n == 0 or not records:
        return {"sampled": 0, "checked_segments": 0, "unsupported": 0, "disagreements": 0}

    rng = random.Random(seed)
    pool = [r for r in records if (r.get("sources") or r.get("answer"))]
    sample = pool if len(pool) <= n else rng.sample(list(pool), n)

    total_segs = 0
    unsupported = 0
    all_disagreements: list[dict[str, Any]] = []

    for rec in sample:
        sources = rec.get("sources") or []
        by_id = {
            str(s.get("chunk_id") or ""): type("C", (), {"text": s.get("text") or ""})()
            for s in sources
            if s.get("chunk_id")
        }
        # Reconstruct segments from answer paragraphs when structured segs absent.
        answer = str(rec.get("answer") or "")
        segs: list[AnswerSegment] = []
        if by_id:
            # One claim per cited source paragraph (best-effort for eval sample).
            paras = [p.strip() for p in re.split(r"\n\s*\n", answer) if p.strip()]
            ids = list(by_id.keys())
            for i, para in enumerate(paras or [answer]):
                # Strip trailing citation chips roughly.
                claim = re.sub(r"\[[^\]]+§[^\]]+\]\s*$", "", para).strip()
                cid = ids[min(i, len(ids) - 1)]
                if claim:
                    segs.append(AnswerSegment(text=claim, citation_chunk_id=cid))
        if not segs:
            continue
        report = audit_segments_semantic(
            question=str(rec.get("question") or ""),
            segments=segs,
            chunks_by_id=by_id,
            llm=llm,
            use_llm=True,
        )
        total_segs += len(report.results)
        unsupported += sum(1 for r in report.results if not r.supported)
        all_disagreements.extend(report.disagreements)

    log_semantic_disagreements(all_disagreements)
    summary = {
        "sampled": len(sample),
        "checked_segments": total_segs,
        "unsupported": unsupported,
        "disagreements": len(all_disagreements),
        "disagreement_path": str(SEMANTIC_DISAGREEMENTS_PATH),
    }
    logger.info("semantic groundedness eval sample: %s", summary)
    return summary
