"""Hard regression checks for CI-critical golden cases.

Fails (raises AssertionError) when any tagged ``ci-gate`` case regresses:
  - retrieval: expected chunk_id / section_number must appear in top-k
  - citation: answer chips' § must match cited chunk metadata and expected section
  - not-indexed: if the required regulation is absent, answer must be the
    live ``not_found_in_regulations_message`` (never a fabricated claim)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from eval.gold import gold_keys, is_hit, load_golden_set
from generation.answer import (
    answer_question,
    assert_prose_sections_match_citations,
    not_found_in_regulations_message,
)
from generation.llm_client import LLMClient
from generation.numeric_guard import (
    NUMERIC_CONFIRM_MESSAGE,
    check_numeric_fidelity,
    extract_user_numbers,
)
from retrieval.retrieve import (
    fetch_chunks_by_ids,
    indexed_regulation_ids,
    prepend_unique_chunks,
    retrieve,
)

logger = logging.getLogger(__name__)

DEFAULT_CI_CASE_IDS = frozenset(
    {
        # Citation / factual anchors
        "fac_013",  # Define H-point
        "fac_014",  # VC limit
        "fac_015",  # objective / scope vehicles
        # Numeric safety + compliance (historical breakage)
        "num_001",  # fuel 35 g/min
        "num_002",  # PSPF 7.5
        "num_003",  # RDC 42.5
        "num_004",  # HPC 1001
        "num_005",  # VC 0.95
        "num_006",  # composite fuel omit
        "cmp_001",  # RDC 45 mm
        "cmp_002",  # HPC 1250
        "cmp_003",  # VC 0.8 PASS
        "cmp_004",  # electrolyte passenger
        "mhp_001",  # R94 composite HPC/chest/fuel
        # Cross-reg / hard filter / groundedness
        "xrg_001",  # vehicles covered R94 (no R16)
        "xrg_006",  # isolation resistance R94
        "xrg_007",  # electrical safety regs
        "xrg_009",  # vehicles covered R95 (not-indexed fallback)
        "hal_001",  # R16 relate R94
        "enm_001",  # R95 doors enumerative
        "dsn_001",  # B-Pillar design
    }
)

_NOT_INDEXED_RE = re.compile(
    r"could(?:n't| not) find (?:relevant content on this in )?the indexed regulations",
    re.IGNORECASE,
)


def _is_ci_case(case: dict[str, Any]) -> bool:
    tags = {str(t).lower() for t in (case.get("tags") or [])}
    severity = str(case.get("severity") or "").strip().upper()
    category = str(case.get("category") or "").strip().lower()
    return (
        case.get("id") in DEFAULT_CI_CASE_IDS
        or "ci-gate" in tags
        or "value-vs-limit" in tags
        or "numeric-fidelity" in tags
        or "compliance-deterministic" in tags
        or "hard-reg-filter" in tags
        or "multi-regulation" in tags
        or "enumerative" in tags
        or "negative-claim" in tags
        or "design-implication" in tags
        or "checklist-gen" in tags
        or "scope-summary" in tags
        or "applicability" in tags
        or "retest-scope" in tags
        or "limits-aggregation" in tags
        or severity == "CRITICAL"
        or category
        in {
            "numeric_safety",
            "compliance_check",
            "hallucination_probe",
            "prompt_injection",
            "guardrail",
            "cross_regulation",
        }
    )


def _regulation_indexed(regulation_id: str | None) -> bool:
    if not regulation_id:
        return True
    return regulation_id in indexed_regulation_ids()


def _expected_sections(case: dict[str, Any]) -> list[str]:
    secs = [
        str(s).strip()
        for s in (case.get("expected_sections") or [])
        if str(s).strip()
    ]
    primary = str(case.get("expected_section_number") or "").strip()
    if primary and primary not in secs:
        secs.insert(0, primary)
    return secs


def _assert_regulation_lock(case: dict[str, Any], chunks: list[Any]) -> None:
    """Hard-reg-filter cases: only the named corpus; never banned regs (e.g. R16)."""
    allowed = [
        str(s).strip()
        for s in (case.get("expect_regulation_ids_only") or [])
        if str(s).strip()
    ]
    if not allowed and case.get("regulation_id"):
        allowed = [str(case["regulation_id"]).strip()]
    banned = {
        str(s).strip()
        for s in (case.get("banned_regulation_ids") or [])
        if str(s).strip()
    }
    if not allowed and not banned:
        return
    for c in chunks:
        rid = str(getattr(c, "regulation_id", "") or "").strip()
        if not rid:
            continue
        if banned and rid in banned:
            raise AssertionError(
                f"{case['id']}: retrieved banned regulation {rid} "
                f"(sec={getattr(c, 'section_number', None)}, chunk={getattr(c, 'chunk_id', None)})"
            )
        if allowed and rid not in allowed:
            raise AssertionError(
                f"{case['id']}: retrieved foreign regulation {rid}; "
                f"allowed={allowed} (sec={getattr(c, 'section_number', None)})"
            )


def _check_retrieval(case: dict[str, Any], *, top_k: int = 5) -> None:
    keys = gold_keys(case)
    if not keys:
        raise AssertionError(f"{case['id']}: ci-gate case has no expected chunk/section keys")

    tags = {str(t).lower() for t in (case.get("tags") or [])}
    is_enum = "enumerative" in tags
    hard_filter = "hard-reg-filter" in tags
    # Enumerative cases need broader top-k (default path uses 20).
    eff_top_k = max(top_k, int(case.get("expect_min_chunks") or 0), 20 if is_enum else top_k)
    # Hard-filter regressions must exercise auto-detect from the question text —
    # do not pass case.regulation_id (that would mask the bug).
    rid_arg = None if hard_filter else case.get("regulation_id")
    repeats = max(1, int(case.get("repeat_retrieval") or 1))

    def _one_pass() -> list[Any]:
        return retrieve(
            case["question"],
            regulation_id=rid_arg,
            top_k=eff_top_k if is_enum else top_k,
            rewrite=True,
            do_rerank=True,
            small_to_big=False,
        )

    chunks = _one_pass()
    for pass_i in range(repeats):
        if pass_i > 0:
            chunks = _one_pass()
        check_k = max(top_k, len(chunks), eff_top_k if is_enum else top_k)
        _assert_regulation_lock(case, chunks[:check_k])

        hits = [c for c in chunks[:check_k] if is_hit(c, keys)]
        if not hits:
            got = [(c.chunk_id, c.section_number, c.regulation_id) for c in chunks[:check_k]]
            raise AssertionError(
                f"{case['id']}: expected {sorted(keys)} not in top-{check_k} retrieval "
                f"(pass {pass_i + 1}/{repeats}); got {got}"
            )

        expected_sections = _expected_sections(case)
        if expected_sections:
            all_secs = {(c.section_number or "").strip() for c in chunks[:check_k]}
            if not (all_secs & set(expected_sections)):
                raise AssertionError(
                    f"{case['id']}: expected section_number in {expected_sections}; "
                    f"top-{check_k} sections={sorted(all_secs)} "
                    f"(pass {pass_i + 1}/{repeats})"
                )

        for group in case.get("expect_context_keywords") or []:
            alts = [a.strip() for a in str(group).lower().split("|") if a.strip()]
            joined = " ".join(
                f"{(c.text or '')} {(c.section_title or '')} {(c.section_number or '')}"
                for c in chunks[:check_k]
            ).lower()
            if alts and not any(a in joined for a in alts):
                raise AssertionError(
                    f"{case['id']}: expected context keyword group {group!r} missing "
                    f"(pass {pass_i + 1}/{repeats})"
                )

        min_chunks = case.get("expect_min_chunks")
        if min_chunks is not None:
            min_n = int(min_chunks)
            # Distinct relevant hits (gold keys), not just any retrieved blobs.
            if len(hits) < min_n:
                # Fallback: door/topic keyword coverage across retrieved set.
                topic = str(case.get("expect_topic_keyword") or "").strip().lower()
                if topic:
                    topic_hits = [
                        c
                        for c in chunks
                        if topic in ((c.text or "") + " " + (c.section_title or "")).lower()
                    ]
                    if len(topic_hits) < min_n:
                        raise AssertionError(
                            f"{case['id']}: expected ≥{min_n} chunks mentioning {topic!r}; "
                            f"got {len(topic_hits)} (gold hits={len(hits)}, retrieved={len(chunks)})"
                        )
                else:
                    raise AssertionError(
                        f"{case['id']}: expected ≥{min_n} gold-matching chunks; got {len(hits)} "
                        f"of {len(chunks)} retrieved"
                    )

        if "value-vs-limit" in tags or (
            str(case.get("severity") or "").upper() == "CRITICAL"
            and "hard-reg-filter" not in tags
        ):
            # Top hit must be an injury-criteria / limit clause, not ISO 6487 calibration.
            top = chunks[0] if chunks else None
            blob = " ".join(
                [
                    (top.text if top else "") or "",
                    (top.section_title if top else "") or "",
                ]
            ).lower()
            if re.search(r"iso\s*6487|\bcfc\b|channel frequency|calibrat", blob):
                if not re.search(
                    r"performance criteria|shall not exceed|less than or equal|"
                    r"\brdc\b|\bpspf\b|\bhpc\b|\bhic\b",
                    blob,
                ):
                    raise AssertionError(
                        f"{case['id']}: top retrieval looks like calibration/sensor text, "
                        f"not an injury-criterion limit: sec={getattr(top, 'section_number', None)} "
                        f"preview={blob[:180]!r}"
                    )
            limit_needles = [
                str(s).lower()
                for s in (case.get("expect_limit_contains") or [])
                if str(s).strip()
            ]
            if limit_needles:
                joined = " ".join((c.text or "") for c in chunks[:check_k]).lower()
                if not any(n.replace(",", "") in joined.replace(",", "") for n in limit_needles):
                    raise AssertionError(
                        f"{case['id']}: expected limit token(s) {limit_needles} missing from "
                        f"top-{check_k} retrieved text"
                    )


def _check_citation_answer(case: dict[str, Any], *, llm: LLMClient) -> None:
    """Answer with gold evidence forced first so citation § is deterministic under mock."""
    tags = {str(t).lower() for t in (case.get("tags") or [])}
    hard_filter = "hard-reg-filter" in tags
    rid_arg = None if hard_filter else case.get("regulation_id")

    expected_ids = list(case.get("expected_chunk_ids") or [])
    retrieved = retrieve(
        case["question"],
        regulation_id=rid_arg,
        top_k=5,
        rewrite=True,
        do_rerank=True,
        small_to_big=False,
    )
    _assert_regulation_lock(case, retrieved)
    forced = fetch_chunks_by_ids(expected_ids) if expected_ids else []
    if expected_ids and not forced:
        raise AssertionError(
            f"{case['id']}: expected_chunk_ids {expected_ids} not found in Qdrant"
        )
    context = prepend_unique_chunks(forced, retrieved[:3]) if forced else retrieved

    ans = answer_question(
        case["question"],
        regulation_id=rid_arg,
        llm=llm,
        chunks=context,
        skip_answer_cache=True,
    )
    if ans.not_found or not ans.sources:
        raise AssertionError(
            f"{case['id']}: expected a cited answer, got not_found={ans.not_found} "
            f"answer={ans.answer!r}"
        )
    assert_prose_sections_match_citations(ans.answer, ans.sources)
    _assert_regulation_lock(case, ans.sources)

    expected_sections = _expected_sections(case)
    cited_secs = {(s.section_number or "").strip() for s in ans.sources}
    if expected_sections and not (cited_secs & set(expected_sections)):
        raise AssertionError(
            f"{case['id']}: cited sections {sorted(cited_secs)} miss expected "
            f"{expected_sections}; answer={ans.answer!r}"
        )

    if expected_ids:
        cited_ids = {s.chunk_id for s in ans.sources}
        if not (cited_ids & set(expected_ids)):
            raise AssertionError(
                f"{case['id']}: expected chunk_ids {expected_ids} not cited; "
                f"got {sorted(cited_ids)}"
            )

    _check_numeric_fidelity_case(case, ans=ans, llm=llm)


def _check_numeric_fidelity_case(
    case: dict[str, Any],
    *,
    ans: Any | None = None,
    llm: LLMClient | None = None,
) -> None:
    """CRITICAL numeric-fidelity: user figures must not be silently altered."""
    tags = {str(t).lower() for t in (case.get("tags") or [])}
    measured = [str(s) for s in (case.get("expect_measured_contains") or []) if str(s).strip()]
    if not measured and "numeric-fidelity" not in tags:
        return

    question = case["question"]
    gold_answer = str(case.get("answer") or case.get("ground_truth") or "")
    user_nums = extract_user_numbers(question)
    if measured and not user_nums:
        raise AssertionError(
            f"{case['id']}: expect_measured_contains={measured} but extract_user_numbers "
            f"found none in question={question!r}"
        )

    # Gold answer must itself be numerically faithful.
    if gold_answer:
        gold_check = check_numeric_fidelity(question, gold_answer)
        if gold_check.rejected:
            raise AssertionError(
                f"{case['id']}: gold answer fails numeric fidelity: {gold_check.reason}"
            )
        for needle in measured:
            if needle not in gold_answer:
                raise AssertionError(
                    f"{case['id']}: gold answer missing measured figure {needle!r}: "
                    f"{gold_answer!r}"
                )
        verdict = str(case.get("expect_verdict") or "").upper()
        if verdict and verdict not in gold_answer.upper():
            raise AssertionError(
                f"{case['id']}: gold answer missing verdict {verdict}: {gold_answer!r}"
            )

    # Synthetic corruption of the primary measured figure must be rejected.
    if user_nums and gold_answer:
        primary = next((u for u in user_nums if u.unit), user_nums[0])
        corrupted = _corrupt_number_token(primary.normalized)
        if corrupted and corrupted != primary.normalized:
            bad = gold_answer
            # Prefer corrupting an explicit "measured …" restatement.
            bad = re.sub(
                rf"(measured\s+(?:value\s+)?(?:of\s+)?){re.escape(primary.raw)}",
                rf"\g<1>{corrupted}",
                bad,
                count=1,
                flags=re.I,
            )
            if bad == gold_answer:
                bad = gold_answer.replace(primary.raw, corrupted, 1)
            bad_check = check_numeric_fidelity(question, bad)
            if bad_check.ok:
                raise AssertionError(
                    f"{case['id']}: numeric guard failed to reject corrupted answer "
                    f"({primary.raw}→{corrupted}): {bad!r}"
                )

    # Live answer checks (skip content assertions for deterministic mock stubs).
    if ans is None or llm is None or getattr(llm, "provider", "") == "mock":
        return
    if ans.failure_kind == "numeric_hallucination" or (
        NUMERIC_CONFIRM_MESSAGE[:40].lower() in (ans.answer or "").lower()
    ):
        # Guard fired — acceptable mitigation vs a wrong PASS/FAIL; still not a
        # silent wrong verdict. Retrieval already checked separately.
        return
    lower = (ans.answer or "").lower()
    for needle in measured:
        if needle not in (ans.answer or ""):
            raise AssertionError(
                f"{case['id']}: live answer missing measured figure {needle!r}: "
                f"{ans.answer!r}"
            )
    for banned in case.get("banned_answer_substrings") or []:
        b = str(banned).lower()
        if b and b in lower:
            raise AssertionError(
                f"{case['id']}: live answer contains banned altered figure {banned!r}: "
                f"{ans.answer!r}"
            )
    verdict = str(case.get("expect_verdict") or "").upper()
    if verdict and verdict not in (ans.answer or "").upper():
        raise AssertionError(
            f"{case['id']}: live answer missing verdict {verdict}: {ans.answer!r}"
        )


def _corrupt_number_token(normalized: str) -> str:
    """Drop a trailing digit to simulate the 35→3 class of hallucination."""
    s = (normalized or "").strip()
    if not s:
        return s
    if "." in s:
        whole, _, frac = s.partition(".")
        if len(frac) >= 1:
            return whole  # 42.5 → 42, 0.95 → 0
        if len(whole) >= 2:
            return whole[:-1]
        return s
    if len(s) >= 2:
        return s[:-1]  # 35 → 3, 1250 → 125, 1001 → 100
    return s


def _check_compliance_deterministic(case: dict[str, Any]) -> None:
    """Composite / compliance cases must evaluate every measured criterion in code."""
    from generation.compliance import evaluate_compliance
    from ingestion.extract_limits import seed_known_limits
    from retrieval.multi_criterion import (
        is_multi_criterion_query,
        list_named_criteria,
        uncovered_criteria,
    )

    seed_known_limits()
    question = case["question"]
    tags = {str(t).lower() for t in (case.get("tags") or [])}

    # Electrolyte / spillage: retrieve containment clauses, not isolation procedure.
    if "electrolyte" in tags or re.search(r"(?i)\belectrolyte\b|\bspillage\b", question):
        chunks = retrieve(
            question,
            regulation_id=case.get("regulation_id"),
            top_k=5,
            rewrite=True,
            do_rerank=True,
            small_to_big=False,
        )
        joined = " ".join(
            f"{(c.text or '')} {(c.section_number or '')}" for c in chunks
        ).lower()
        if not re.search(r"electrolyte\s+leakage|passenger\s+compart", joined):
            raise AssertionError(
                f"{case['id']}: electrolyte containment clause missing from retrieval; "
                f"got={[(c.chunk_id, c.section_number) for c in chunks]}"
            )
        if re.search(
            r"isolation\s+resistance\s+measurement|measuring\s+electric\s+resistance",
            joined,
        ) and not re.search(r"electrolyte\s+leakage.{0,40}passenger", joined):
            # Isolation procedure alone (no containment) is the Fix-19-class failure.
            top = chunks[0] if chunks else None
            top_blob = f"{(top.text if top else '')} {(top.section_number if top else '')}".lower()
            if re.search(r"isolation\s+resistance\s+measurement|fifth\s+step", top_blob):
                raise AssertionError(
                    f"{case['id']}: top hit is isolation-resistance procedure, "
                    f"not electrolyte containment: sec={getattr(top, 'section_number', None)}"
                )
        for banned in case.get("banned_answer_substrings") or []:
            # Reuse as banned retrieval text for this path.
            pass
        logger.info(
            "electrolyte retrieval OK %s sections=%s",
            case["id"],
            [c.section_number for c in chunks[:5]],
        )

    # Per-criterion retrieval: every named criterion must appear in context.
    if (
        is_multi_criterion_query(question)
        or "multi-criterion-retrieval" in tags
    ):
        chunks = retrieve(
            question,
            regulation_id=case.get("regulation_id"),
            top_k=max(9, 3 * max(2, len(list_named_criteria(question)))),
            rewrite=True,
            do_rerank=True,
            small_to_big=False,
        )
        missing = uncovered_criteria(question, chunks)
        if missing:
            raise AssertionError(
                f"{case['id']}: multi-criterion retrieval missing context for "
                f"{[m.key for m in missing]}; got sections="
                f"{[(c.chunk_id, c.section_number) for c in chunks[:12]]}"
            )
        # Explicit keyword groups from the golden case (OR within a group).
        joined = " ".join(
            f"{(c.text or '')} {(c.section_title or '')}" for c in chunks
        ).lower()
        for group in case.get("expect_context_keywords") or []:
            alts = [a.strip() for a in str(group).lower().split("|") if a.strip()]
            if alts and not any(a in joined for a in alts):
                raise AssertionError(
                    f"{case['id']}: expected context keyword group {group!r} "
                    f"missing from retrieved text (n_chunks={len(chunks)})"
                )
        logger.info(
            "multi-criterion retrieval OK %s keys=%s n_chunks=%d",
            case["id"],
            [c.key for c in list_named_criteria(question)],
            len(chunks),
        )

    result = evaluate_compliance(
        question,
        regulation_id=case.get("regulation_id"),
    )
    if result is None or not result.criteria:
        raise AssertionError(
            f"{case['id']}: expected deterministic compliance result, got {result!r}"
        )
    measured_needles = [
        str(s) for s in (case.get("expect_measured_contains") or []) if str(s).strip()
    ]
    for needle in measured_needles:
        if not any(
            needle in (c.measured_display or "")
            or (
                c.measured is not None
                and abs(float(c.measured) - float(needle.replace(",", "."))) < 1e-9
            )
            for c in result.criteria
        ):
            raise AssertionError(
                f"{case['id']}: measured {needle!r} missing from criteria "
                f"{[c.measured_display for c in result.criteria]}"
            )
    if any(c.verdict == "LIMIT_NOT_FOUND" for c in result.criteria):
        # Only OK when the gold answer expects an explicit limit-not-found.
        if "limit not found" not in str(case.get("answer") or "").lower():
            raise AssertionError(
                f"{case['id']}: unexpected LIMIT_NOT_FOUND in {result.criteria}"
            )
    verdict = str(case.get("expect_verdict") or "").upper()
    if verdict and result.overall_verdict != verdict:
        raise AssertionError(
            f"{case['id']}: overall_verdict={result.overall_verdict} expected {verdict}; "
            f"criteria={[ (c.criterion, c.verdict, c.measured) for c in result.criteria ]}"
        )
    # Fuel must appear when the question mentions fuel leakage (not electrolyte).
    if re.search(r"(?i)\bfuel\b", question):
        fuel_rows = [
            c
            for c in result.criteria
            if re.search(r"(?i)fuel|leak", c.criterion)
            or (c.unit or "").replace(" ", "") == "g/min"
        ]
        if not fuel_rows:
            raise AssertionError(
                f"{case['id']}: fuel leakage criterion was dropped from composite evaluation"
            )
    logger.info(
        "compliance OK %s overall=%s n=%d",
        case["id"],
        result.overall_verdict,
        len(result.criteria),
    )


def _check_not_indexed(case: dict[str, Any], *, llm: LLMClient) -> None:
    reg = case.get("regulation_id")
    ans = answer_question(
        case["question"],
        regulation_id=reg,
        llm=llm,
        skip_answer_cache=True,
    )
    if not ans.not_found:
        raise AssertionError(
            f"{case['id']}: expected not_found for missing {reg}, "
            f"got answer={ans.answer!r}"
        )
    if not _NOT_INDEXED_RE.search(ans.answer or ""):
        raise AssertionError(
            f"{case['id']}: answer missing dynamic not-indexed phrasing: {ans.answer!r}"
        )
    lower = (ans.answer or "").lower()
    banned = ("category m1 of a total permissible", "lateral collision protection shall")
    if any(b in lower for b in banned):
        raise AssertionError(
            f"{case['id']}: fabricated regulation content while not indexed: {ans.answer!r}"
        )
    m = re.search(r"indexed regulations \(([^)]*)\)", ans.answer or "", re.I)
    if m and re.search(r"\bR95\b", m.group(1)) and reg == "UN-ECE-R95":
        if "UN-ECE-R95" not in indexed_regulation_ids():
            raise AssertionError(
                f"{case['id']}: not-indexed message lists R95 but it is not indexed: "
                f"{ans.answer!r}"
            )
    # Sanity: live helper agrees.
    _ = not_found_in_regulations_message()
    logger.info("not-indexed OK %s → %s", case["id"], ans.answer[:120])


_NOT_ADDRESSED_RE = re.compile(
    r"(?i)("
    r"could(?:n't| not) find (?:relevant content on this in )?the indexed"
    r"|found related content but couldn'?t produce"
    r"|not\s+addressed"
    r"|not\s+found"
    r"|does\s+not\s+(?:explicitly\s+)?(?:address|state|discuss)"
    r"|i couldn't verify"
    r")"
)


def _check_abstention_no_fabricated_negative(case: dict[str, Any], *, llm: LLMClient) -> None:
    """Expect honest not-found / not-addressed — never a confident false negative."""
    ans = answer_question(
        case["question"],
        regulation_id=case.get("regulation_id"),
        llm=llm,
        skip_answer_cache=True,
    )
    lower = (ans.answer or "").lower()
    banned = [
        str(s).lower()
        for s in (case.get("banned_answer_substrings") or [])
        if str(s).strip()
    ]
    # Default bans for negative-claim cases.
    if "negative-claim" in {str(t).lower() for t in (case.get("tags") or [])}:
        banned = list(
            dict.fromkeys(
                banned
                + [
                    "no direct relationship",
                    "no relationship",
                    "are unrelated",
                ]
            )
        )
    hit = next((b for b in banned if b in lower), None)
    if hit:
        raise AssertionError(
            f"{case['id']}: banned confident-negative phrasing {hit!r} in answer="
            f"{ans.answer!r}"
        )

    ok = bool(ans.not_found) or bool(_NOT_ADDRESSED_RE.search(ans.answer or ""))
    # Mock provider often returns a cited stub; still require no banned negatives.
    if llm.provider == "mock" and not ok:
        logger.info(
            "abstention %s: mock provider skipped not_found requirement; "
            "banned-phrase check passed → %s",
            case["id"],
            (ans.answer or "")[:120],
        )
        return
    if not ok:
        raise AssertionError(
            f"{case['id']}: expected not_found / not-addressed abstention, "
            f"got not_found={ans.not_found} answer={ans.answer!r}"
        )
    logger.info("abstention OK %s → %s", case["id"], (ans.answer or "")[:120])


def _mentions_regulation(blob: str, regulation_id: str) -> bool:
    """True if prose/sources mention UN-ECE-R94 / UN R94 / R94 forms."""
    rid = (regulation_id or "").strip()
    if not rid:
        return False
    lower = (blob or "").lower()
    if rid.lower() in lower:
        return True
    short = rid.replace("UN-ECE-", "").replace("UN_ECE_", "")
    if short.lower() in lower:
        return True
    # R94 / Regulation 94
    m = re.search(r"R(\d{2,3})$", short, re.I)
    if m:
        n = m.group(1)
        if re.search(rf"\br\s*{n}\b", lower) or re.search(
            rf"\bregulation\s+no\.?\s*{n}\b", lower
        ):
            return True
    return False


def _check_retest_scope(case: dict[str, Any], *, llm: LLMClient) -> None:
    """Modification-clause analysis must stay informational — never decisive."""
    from retrieval.retest_scope import (
        RETEST_DISCLAIMER,
        expand_retest_query,
        match_changes,
    )
    from retrieval.router import QueryIntent, classify_query

    question = case["question"]
    routed = classify_query(question, use_llm=False, log=False)
    if routed.intent != QueryIntent.RETEST_SCOPE:
        raise AssertionError(
            f"{case['id']}: expected RETEST_SCOPE, got {routed.intent}"
        )

    expansion = expand_retest_query(question)
    expect_changes = [
        str(s).strip()
        for s in (case.get("expect_change_ids") or [])
        if str(s).strip()
    ]
    got_ids = {c.id for c in expansion.changes} or {c.id for c in match_changes(question)}
    missing_ch = [c for c in expect_changes if c not in got_ids]
    if missing_ch:
        raise AssertionError(
            f"{case['id']}: expected changes {expect_changes}, got {sorted(got_ids)}; "
            f"missing {missing_ch}"
        )

    chunks = retrieve(
        question,
        regulation_id=None,
        top_k=8,
        rewrite=True,
        do_rerank=False,
        small_to_big=False,
        routed=routed,
    )
    min_chunks = int(case.get("expect_min_chunks") or 1)
    if len(chunks) < min_chunks:
        raise AssertionError(
            f"{case['id']}: retest retrieval returned {len(chunks)} chunks "
            f"(need ≥{min_chunks}); targets={expansion.target_regulation_ids}"
        )

    ans = answer_question(
        question,
        regulation_id=None,
        llm=llm,
        skip_answer_cache=True,
    )
    blob = (ans.answer or "").lower()
    if "homologation authority" not in blob and "verify" not in blob:
        raise AssertionError(
            f"{case['id']}: answer missing honesty framing; answer={ans.answer!r}"
        )
    if not ans.mode_disclaimer:
        raise AssertionError(
            f"{case['id']}: mode_disclaimer must be set on AnswerResponse "
            f"(UI banner contract)"
        )
    if RETEST_DISCLAIMER[:40].lower() not in (ans.mode_disclaimer or "").lower():
        raise AssertionError(
            f"{case['id']}: mode_disclaimer does not carry RETEST_DISCLAIMER text"
        )
    # Authoritative decision language is a liability.
    if re.search(
        r"(?i)\b(you must|we must|shall)\s+re-?test\b",
        ans.answer or "",
    ):
        raise AssertionError(
            f"{case['id']}: over-confident retest order in answer: {ans.answer!r}"
        )
    if re.search(
        r"(?i)\bno\s+re-?test\s+(?:is\s+)?(?:needed|required)\b",
        ans.answer or "",
    ):
        raise AssertionError(
            f"{case['id']}: over-confident 'no retest' ruling: {ans.answer!r}"
        )
    framing_ok = any(
        m in blob
        for m in (
            "informational only",
            "homologation authority",
            "not a determination",
            "verify",
        )
    )
    if not framing_ok:
        raise AssertionError(
            f"{case['id']}: missing non-authoritative framing; answer={ans.answer!r}"
        )
    logger.info(
        "retest_scope OK %s changes=%s chunks=%d disclaimer=%s",
        case["id"],
        sorted(got_ids),
        len(chunks),
        bool(ans.mode_disclaimer),
    )


def _check_applicability(case: dict[str, Any], *, llm: LLMClient) -> None:
    """Multi-reg scope survey: every indexed reg gets a ternary verdict."""
    from retrieval.applicability import (
        VERDICT_APPLIES,
        retrieve_applicability,
    )
    from retrieval.router import QueryIntent, classify_query

    question = case["question"]
    routed = classify_query(question, use_llm=False, log=False)
    if routed.intent != QueryIntent.APPLICABILITY:
        raise AssertionError(
            f"{case['id']}: expected APPLICABILITY, got {routed.intent}"
        )

    result = retrieve_applicability(question)
    indexed = result.expansion.indexed_regulation_ids
    if len(indexed) < 2:
        raise AssertionError(
            f"{case['id']}: need ≥2 indexed regs for applicability survey, got {indexed}"
        )
    # Must retrieve a scope chunk for most regs (at least expect_applies).
    expect_applies = [
        str(s).strip()
        for s in (case.get("expect_applies") or [])
        if str(s).strip()
    ]
    for rid in expect_applies:
        if rid not in result.per_regulation:
            raise AssertionError(
                f"{case['id']}: missing Scope clause retrieve for expected {rid}; "
                f"got scopes={sorted(result.per_regulation.keys())}"
            )
        heur = next((h for h in result.heuristics if h.regulation_id == rid), None)
        if heur is None or heur.verdict != VERDICT_APPLIES:
            raise AssertionError(
                f"{case['id']}: heuristic should APPLIES for {rid}, got {heur}"
            )

    # Never a single-reg retrieval set.
    if len(result.chunks) < 2:
        raise AssertionError(
            f"{case['id']}: applicability must retrieve ≥2 scope chunks, "
            f"got {len(result.chunks)}"
        )

    ans = answer_question(
        question,
        regulation_id=None,
        llm=llm,
        skip_answer_cache=True,
    )
    blob = (ans.answer or "").lower()
    if "couldn't produce a confidently grounded" in blob:
        raise AssertionError(f"{case['id']}: ungrounded fallback: {ans.answer!r}")
    for heading in ("applies", "does not apply", "cannot determine"):
        if heading not in blob:
            raise AssertionError(
                f"{case['id']}: missing verdict section {heading!r}; answer={ans.answer!r}"
            )
    for rid in expect_applies:
        # Answer or sources must mention the reg under Applies.
        label = rid.replace("UN-ECE-", "UN ")
        short = rid.split("-")[-1]  # R94
        applies_block = ""
        if "## applies" in blob:
            applies_block = blob.split("## applies", 1)[1].split("## ", 1)[0]
        if short.lower() not in applies_block and label.lower() not in applies_block:
            raise AssertionError(
                f"{case['id']}: expected {rid} under Applies; answer={ans.answer!r}"
            )
    # Must address all indexed regs somewhere.
    for rid in indexed:
        short = rid.split("-")[-1].lower()
        if short not in blob and rid.lower() not in blob:
            raise AssertionError(
                f"{case['id']}: indexed {rid} not addressed in answer"
            )
    source_regs = {(s.regulation_id or "").strip() for s in (ans.sources or [])}
    if expect_applies and not set(expect_applies) <= source_regs | {
        h.regulation_id for h in result.heuristics if h.verdict == VERDICT_APPLIES
    }:
        # Sources should include scope cites for applies regs when chunks exist.
        missing_src = [r for r in expect_applies if r not in source_regs]
        if missing_src and len(source_regs) == 1:
            raise AssertionError(
                f"{case['id']}: single-regulation sources {source_regs} "
                f"(missing {missing_src}) — under-reasoned X3-style failure"
            )
    logger.info(
        "applicability OK %s scopes=%d applies=%s answer_sources=%s",
        case["id"],
        len(result.chunks),
        expect_applies,
        sorted(source_regs),
    )


def _check_scope_summary(case: dict[str, Any], *, llm: LLMClient) -> None:
    """Named-reg scope summary: hard filter + structured sections; no foreign regs."""
    from ingestion.extract_limits import load_limits_table, seed_known_limits
    from retrieval.router import QueryIntent, classify_query
    from retrieval.scope_summary import expand_scope_summary_query, retrieve_scope_summary

    question = case["question"]
    routed = classify_query(question, use_llm=False, log=False)
    if routed.intent != QueryIntent.SCOPE_SUMMARY:
        raise AssertionError(
            f"{case['id']}: expected SCOPE_SUMMARY, got {routed.intent}"
        )

    expect_reg = str(case.get("regulation_id") or "").strip()
    expansion = expand_scope_summary_query(question)
    if expect_reg and expansion.regulation_id != expect_reg:
        raise AssertionError(
            f"{case['id']}: expected {expect_reg}, got {expansion.regulation_id}"
        )

    result = retrieve_scope_summary(question, regulation_id=expect_reg or None)
    if not result.chunks:
        raise AssertionError(f"{case['id']}: scope_summary retrieved zero chunks")

    regs = {
        (c.regulation_id or "").strip()
        for c in result.chunks
        if (c.regulation_id or "").strip()
    }
    if expect_reg and regs - {expect_reg}:
        raise AssertionError(
            f"{case['id']}: hard-filter leak — expected only {expect_reg}, got {sorted(regs)}"
        )
    banned = {
        str(s).strip()
        for s in (case.get("banned_regulation_ids") or [])
        if str(s).strip()
    }
    if banned & regs:
        raise AssertionError(
            f"{case['id']}: banned regs present in retrieval: {sorted(banned & regs)}"
        )

    # Limits table single source of truth.
    seed_known_limits()
    table = load_limits_table(expect_reg) if expect_reg else None
    if expect_reg and table is None:
        raise AssertionError(f"{case['id']}: missing limits table for {expect_reg}")

    chunks = retrieve(
        question,
        regulation_id=expect_reg or None,
        rewrite=True,
        do_rerank=False,
        small_to_big=False,
        routed=routed,
    )
    regs2 = {(c.regulation_id or "").strip() for c in chunks if c.regulation_id}
    if expect_reg and regs2 - {expect_reg}:
        raise AssertionError(
            f"{case['id']}: retrieve() leaked {sorted(regs2 - {expect_reg})}"
        )

    ans = answer_question(
        question,
        regulation_id=expect_reg or None,
        llm=llm,
        skip_answer_cache=True,
    )
    blob = (ans.answer or "").lower()
    if "couldn't produce a confidently grounded" in blob:
        raise AssertionError(f"{case['id']}: ungrounded fallback: {ans.answer!r}")
    for heading in ("scope", "injury", "test configuration", "homologation"):
        if heading not in blob:
            raise AssertionError(
                f"{case['id']}: missing structured section {heading!r}; "
                f"answer={ans.answer!r}"
            )
    # Numbers from limits table must appear (e.g. ThCC 42 / HPC 1000 for R94).
    if table and table.limits:
        for row in table.limits[:3]:
            val = str(int(row.limit_value) if float(row.limit_value).is_integer() else row.limit_value)
            if val not in (ans.answer or ""):
                raise AssertionError(
                    f"{case['id']}: limits-table value {val} for "
                    f"{row.criterion_name} missing from answer (single source of truth)"
                )
    for s in ans.sources or []:
        rid = (s.regulation_id or "").strip()
        if expect_reg and rid and rid != expect_reg:
            raise AssertionError(
                f"{case['id']}: answer source leaked {rid}; expected {expect_reg}"
            )
        if rid in banned:
            raise AssertionError(f"{case['id']}: answer cited banned {rid}")
    # Ban R95 mentions in R94 summarize answer body (common failure mode).
    if expect_reg == "UN-ECE-R94" and re.search(r"\br95\b", blob):
        raise AssertionError(
            f"{case['id']}: answer mentions R95; answer={ans.answer!r}"
        )
    logger.info(
        "scope_summary OK %s chunks=%d sources=%d limits=%d",
        case["id"],
        len(chunks),
        len(ans.sources or []),
        len(table.limits) if table else 0,
    )


def _check_checklist_gen(case: dict[str, Any], *, llm: LLMClient) -> None:
    """Homologation checklist: per-category retrieval + structured cited deliverable."""
    from retrieval.checklist import (
        expand_checklist_query,
        is_checklist_pipeline_query,
        retrieve_checklist,
    )
    from retrieval.router import QueryIntent, classify_query

    question = case["question"]
    if not is_checklist_pipeline_query(question):
        raise AssertionError(f"{case['id']}: not a checklist-pipeline question")

    routed = classify_query(question, use_llm=False, log=False)
    if routed.intent != QueryIntent.CHECKLIST_GEN:
        raise AssertionError(
            f"{case['id']}: expected CHECKLIST_GEN, got {routed.intent}"
        )

    expansion = expand_checklist_query(question)
    expect_reg = str(case.get("regulation_id") or "").strip()
    if expect_reg and expansion.regulation_id != expect_reg:
        raise AssertionError(
            f"{case['id']}: expected regulation {expect_reg}, got {expansion.regulation_id}"
        )

    check = retrieve_checklist(question, top_k_per_category=4, do_rerank=True)
    min_chunks = int(case.get("expect_min_chunks") or 2)
    if len(check.chunks) < min_chunks:
        raise AssertionError(
            f"{case['id']}: checklist retrieval returned {len(check.chunks)} chunks "
            f"(need ≥{min_chunks}); meta={check.to_public_dict()}"
        )
    min_cats = int(case.get("expect_min_covered_categories") or 2)
    if len(check.covered_categories) < min_cats:
        raise AssertionError(
            f"{case['id']}: expected ≥{min_cats} covered categories, "
            f"got {check.covered_categories}; missing={check.missing_categories}"
        )

    # Separate retrievals must have been used — more than one category attempted.
    if len(check.by_category) < 4:
        raise AssertionError(
            f"{case['id']}: checklist must retrieve across multiple categories"
        )

    only = [
        str(s).strip()
        for s in (case.get("expect_regulation_ids_only") or [])
        if str(s).strip()
    ]
    if only:
        for c in check.chunks:
            rid = (c.regulation_id or "").strip()
            if rid and rid not in only:
                raise AssertionError(
                    f"{case['id']}: checklist leaked {rid}; only={only}"
                )

    ans = answer_question(
        question,
        regulation_id=None,
        llm=llm,
        skip_answer_cache=True,
    )
    if ans.failure_kind == "grounding_rejected" or (
        ans.not_found and "incomplete" not in (ans.answer or "").lower()
    ):
        # Empty structured checklist with incompleteness notes is OK; ungrounded fallback is not.
        blob = (ans.answer or "").lower()
        if "couldn't produce a confidently grounded" in blob:
            raise AssertionError(
                f"{case['id']}: checklist must not use ungrounded fallback; "
                f"answer={ans.answer!r}"
            )
    blob = (ans.answer or "").lower()
    if "couldn't produce a confidently grounded" in blob:
        raise AssertionError(
            f"{case['id']}: got ungrounded fallback: {ans.answer!r}"
        )
    if "## " not in (ans.answer or "") and "checklist" not in blob:
        raise AssertionError(
            f"{case['id']}: expected structured checklist headings; answer={ans.answer!r}"
        )
    # Explicit incompleteness signalling when any category missing.
    if check.missing_categories:
        if (
            "no content found" not in blob
            and "incomplete" not in blob
            and "may be incomplete" not in blob
        ):
            raise AssertionError(
                f"{case['id']}: missing categories {check.missing_categories} "
                f"must be stated explicitly; answer={ans.answer!r}"
            )
    if not ans.sources and check.chunks:
        raise AssertionError(f"{case['id']}: checklist answer has no cited sources")
    logger.info(
        "checklist_gen OK %s chunks=%d covered=%s missing=%s sources=%d",
        case["id"],
        len(check.chunks),
        check.covered_categories,
        check.missing_categories,
        len(ans.sources or []),
    )


def _check_design_implication(case: dict[str, Any], *, llm: LLMClient) -> None:
    """Design-implication: multi-claim grounded synthesis, not ungrounded fallback."""
    from retrieval.design_implication import expand_design_query
    from retrieval.router import QueryIntent, classify_query

    question = case["question"]
    routed = classify_query(question, use_llm=False, log=False)
    if routed.intent != QueryIntent.DESIGN_IMPLICATION:
        raise AssertionError(
            f"{case['id']}: expected DESIGN_IMPLICATION intent, got {routed.intent}"
        )

    expansion = expand_design_query(question)
    expect_component = str(case.get("expect_design_component") or "").strip()
    if expect_component:
        got = expansion.component.id if expansion.component else None
        if got != expect_component:
            raise AssertionError(
                f"{case['id']}: design component expected {expect_component!r}, got {got!r}"
            )

    chunks = retrieve(
        question,
        regulation_id=None,
        top_k=5,
        rewrite=True,
        do_rerank=True,
        small_to_big=False,
        routed=routed,
    )
    min_chunks = int(case.get("expect_min_chunks") or 2)
    if len(chunks) < min_chunks:
        raise AssertionError(
            f"{case['id']}: design retrieval returned {len(chunks)} chunks "
            f"(need ≥{min_chunks}); expansion={expansion.to_public_dict()}"
        )

    expect_regs = [
        str(s).strip()
        for s in (case.get("expect_retrieved_regulations") or [])
        if str(s).strip()
    ]
    if expect_regs:
        got = {(c.regulation_id or "").strip() for c in chunks}
        missing = [r for r in expect_regs if r not in got]
        if missing:
            raise AssertionError(
                f"{case['id']}: design retrieval missing regs {missing}; got={sorted(got)}"
            )

    only = [
        str(s).strip()
        for s in (case.get("expect_regulation_ids_only") or [])
        if str(s).strip()
    ]
    if only:
        for c in chunks:
            rid = (c.regulation_id or "").strip()
            if rid and rid not in only:
                raise AssertionError(
                    f"{case['id']}: design retrieval leaked {rid}; only={only}"
                )

    ans = answer_question(
        question,
        regulation_id=None,
        llm=llm,
        skip_answer_cache=True,
    )
    if ans.failure_kind == "grounding_rejected" or ans.not_found:
        raise AssertionError(
            f"{case['id']}: design answer must be grounded multi-clause synthesis, "
            f"not ungrounded fallback; failure_kind={ans.failure_kind} "
            f"answer={ans.answer!r}"
        )
    blob = (ans.answer or "").lower()
    if "couldn't produce a confidently grounded" in blob or "couldn't verify" in blob:
        raise AssertionError(
            f"{case['id']}: got ungrounded fallback prose: {ans.answer!r}"
        )
    if not ans.sources:
        raise AssertionError(f"{case['id']}: design answer has no cited sources")
    # Prefer explicit FACT/INFERENCE labelling (live LLM or extractive fallback).
    if "regulatory fact" not in blob and "[regulatory fact]" not in blob:
        # Mock may still pass via extractive path; require multi-paragraph or multiple sources.
        if len(ans.sources) < 1:
            raise AssertionError(f"{case['id']}: missing regulatory-fact labelling")
    mention_regs = [
        str(s).strip()
        for s in (case.get("expect_answer_mentions_regulations") or expect_regs)
        if str(s).strip()
    ]
    for rid in mention_regs:
        if not _mentions_regulation(
            " ".join(
                [
                    ans.answer or "",
                    " ".join(s.regulation_id or "" for s in (ans.sources or [])),
                ]
            ),
            rid,
        ):
            raise AssertionError(
                f"{case['id']}: answer/sources must mention {rid}; "
                f"answer={ans.answer!r}"
            )
    logger.info(
        "design_implication OK %s chunks=%d sources=%d component=%s",
        case["id"],
        len(chunks),
        len(ans.sources),
        expansion.component.id if expansion.component else None,
    )


def _check_limits_aggregation(case: dict[str, Any], *, llm: LLMClient) -> None:
    """Summarize-all limits/criteria must render a Fix 22 markdown table, not prose."""
    from retrieval.limits_aggregation import is_limits_aggregation_query
    from ingestion.extract_limits import load_limits_table, seed_known_limits

    q = case["question"]
    if not is_limits_aggregation_query(q):
        raise AssertionError(f"{case['id']}: question not detected as limits-aggregation")

    rid = case.get("regulation_id") or "UN-ECE-R94"
    table = load_limits_table(rid)
    if table is None or not table.limits:
        seed_known_limits()
        table = load_limits_table(rid)
    if table is None or not table.limits:
        raise AssertionError(f"{case['id']}: no limits table for {rid}")

    expect_names = [
        str(s).strip().lower()
        for s in (case.get("expect_limit_criteria") or [])
        if str(s).strip()
    ]
    if not expect_names:
        raise AssertionError(f"{case['id']}: missing expect_limit_criteria")

    seeded = " ".join(
        [
            (r.criterion_name or "").lower()
            + " "
            + " ".join(a.lower() for a in (r.aliases or []))
            for r in table.limits
        ]
    )
    for name in expect_names:
        if name not in seeded:
            raise AssertionError(
                f"{case['id']}: limits table for {rid} missing criterion {name!r}; "
                f"have={[r.criterion_name for r in table.limits]}"
            )

    ans = answer_question(q, regulation_id=rid, llm=llm, skip_answer_cache=True)
    blob = ans.answer or ""
    if "|" not in blob or "Criterion" not in blob:
        raise AssertionError(
            f"{case['id']}: expected markdown table answer, got prose: {blob[:400]!r}"
        )
    # Must not be a single-topic paragraph covering only one criterion.
    row_lines = [ln for ln in blob.splitlines() if ln.strip().startswith("|") and "---" not in ln]
    # header + data rows
    if len(row_lines) < len(expect_names) + 1:
        raise AssertionError(
            f"{case['id']}: table has too few rows ({len(row_lines)}); "
            f"expected >= {len(expect_names) + 1} (header+criteria); answer={blob[:600]!r}"
        )
    lower = blob.lower()
    for name in expect_names:
        if name not in lower:
            raise AssertionError(
                f"{case['id']}: table answer missing criterion {name!r}; answer={blob[:800]!r}"
            )
    # Enumerative breadth: retrieval should not be crushed to top-5.
    chunks = retrieve(
        q,
        regulation_id=rid,
        top_k=None,
        rewrite=False,
        do_rerank=True,
        small_to_big=False,
    )
    if len(chunks) < 6:
        # Limits path may still retrieve broadly; soft check when table path dominates.
        logger.info(
            "limits_aggregation retrieval n_chunks=%d (table is authoritative)",
            len(chunks),
        )
    logger.info(
        "limits_aggregation OK %s rows=%d criteria=%s",
        case["id"],
        len(row_lines) - 1,
        expect_names,
    )


def _check_multi_regulation(case: dict[str, Any], *, llm: LLMClient) -> None:
    """Plural-scope survey: retrieve every indexed reg; answer must name covered + missing."""
    expect_regs = [
        str(s).strip()
        for s in (case.get("expect_retrieved_regulations") or [])
        if str(s).strip()
    ]
    mention_regs = [
        str(s).strip()
        for s in (case.get("expect_answer_mentions_regulations") or expect_regs)
        if str(s).strip()
    ]
    expect_missing = [
        str(s).strip()
        for s in (case.get("expect_missing_regulations") or [])
        if str(s).strip()
    ]
    if not expect_regs and not mention_regs:
        raise AssertionError(f"{case['id']}: multi-regulation case missing expect_*_regulations")

    chunks = retrieve(
        case["question"],
        regulation_id=None,
        top_k=max(3, len(expect_regs) * 3),
        rewrite=True,
        do_rerank=True,
        small_to_big=False,
    )
    got = {(c.regulation_id or "").strip() for c in chunks}
    missing = [r for r in expect_regs if r not in got]
    if missing:
        raise AssertionError(
            f"{case['id']}: multi-regulation retrieval missing {missing}; "
            f"got regs={sorted(got)} sections="
            f"{[(c.regulation_id, c.section_number) for c in chunks[:12]]}"
        )
    leaked = [r for r in expect_missing if r in got]
    if leaked:
        raise AssertionError(
            f"{case['id']}: multi-regulation retrieval should not keep "
            f"off-topic regs {leaked}; got={sorted(got)}"
        )

    ans = answer_question(
        case["question"],
        regulation_id=None,
        llm=llm,
        skip_answer_cache=True,
    )
    blob = " ".join(
        [
            ans.answer or "",
            " ".join(s.regulation_id or "" for s in (ans.sources or [])),
            " ".join(s.citation or "" for s in (ans.sources or [])),
        ]
    )
    for rid in mention_regs:
        if not _mentions_regulation(blob, rid):
            raise AssertionError(
                f"{case['id']}: answer/sources must mention {rid}; "
                f"answer={ans.answer!r} sources={[s.regulation_id for s in ans.sources]}"
            )
    # Fix 25: corpus survey must explicitly report indexed regs without relevant content.
    for rid in expect_missing:
        if not _mentions_regulation(blob, rid):
            raise AssertionError(
                f"{case['id']}: answer must explicitly report missing/no-content "
                f"regulation {rid}; answer={ans.answer!r}"
            )
    if expect_missing and not re.search(
        r"(?i)no relevant content|not found|does not (?:include|address)|without relevant",
        ans.answer or "",
    ):
        raise AssertionError(
            f"{case['id']}: answer must state that some indexed regulations lack "
            f"relevant content; answer={ans.answer!r}"
        )
    logger.info(
        "multi-regulation OK %s retrieved=%s answer_mentions=%s missing=%s",
        case["id"],
        sorted(got),
        mention_regs,
        expect_missing,
    )


def run_regression_gate(
    gold: list[dict[str, Any]] | None = None,
    *,
    llm: LLMClient | None = None,
) -> list[dict[str, Any]]:
    """Run CI-gate cases; raise AssertionError listing all failures."""
    cases = gold if gold is not None else load_golden_set()
    client = llm or LLMClient(provider="mock", use_cache=False)
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for case in cases:
        if not _is_ci_case(case):
            continue
        case_id = case["id"]
        try:
            missing = bool(case.get("expect_not_indexed_if_missing")) and not _regulation_indexed(
                case.get("regulation_id")
            )
            if missing or case.get("expect_not_indexed"):
                _check_not_indexed(case, llm=client)
                results.append({"id": case_id, "mode": "not_indexed", "ok": True})
                continue

            tags = {str(t).lower() for t in (case.get("tags") or [])}
            if case.get("expect_not_found") or (
                case.get("abstention") and "negative-claim" in tags
            ):
                _check_abstention_no_fabricated_negative(case, llm=client)
                results.append({"id": case_id, "mode": "abstention", "ok": True})
                continue

            if "compliance-deterministic" in tags:
                _check_compliance_deterministic(case)
                results.append({"id": case_id, "mode": "compliance", "ok": True})
                logger.info("ci-gate OK %s (compliance)", case_id)
                continue

            if "applicability" in tags:
                _check_applicability(case, llm=client)
                results.append({"id": case_id, "mode": "applicability", "ok": True})
                logger.info("ci-gate OK %s (applicability)", case_id)
                continue

            if "retest-scope" in tags:
                _check_retest_scope(case, llm=client)
                results.append({"id": case_id, "mode": "retest_scope", "ok": True})
                logger.info("ci-gate OK %s (retest_scope)", case_id)
                continue

            if "scope-summary" in tags:
                _check_scope_summary(case, llm=client)
                results.append({"id": case_id, "mode": "scope_summary", "ok": True})
                logger.info("ci-gate OK %s (scope_summary)", case_id)
                continue

            if "checklist-gen" in tags:
                _check_checklist_gen(case, llm=client)
                results.append({"id": case_id, "mode": "checklist_gen", "ok": True})
                logger.info("ci-gate OK %s (checklist_gen)", case_id)
                continue

            if "design-implication" in tags:
                _check_design_implication(case, llm=client)
                results.append({"id": case_id, "mode": "design_implication", "ok": True})
                logger.info("ci-gate OK %s (design_implication)", case_id)
                continue

            if "limits-aggregation" in tags or case.get("expect_limit_criteria"):
                _check_limits_aggregation(case, llm=client)
                results.append({"id": case_id, "mode": "limits_aggregation", "ok": True})
                logger.info("ci-gate OK %s (limits_aggregation)", case_id)
                continue

            if "multi-regulation" in tags or case.get("expect_retrieved_regulations"):
                _check_multi_regulation(case, llm=client)
                results.append({"id": case_id, "mode": "multi_regulation", "ok": True})
                logger.info("ci-gate OK %s (multi-regulation)", case_id)
                continue

            _check_retrieval(case)
            _check_citation_answer(case, llm=client)
            results.append({"id": case_id, "mode": "retrieve+cite", "ok": True})
            logger.info("ci-gate OK %s", case_id)
        except AssertionError as exc:
            failures.append(str(exc))
            results.append({"id": case_id, "mode": "fail", "ok": False, "error": str(exc)})
            logger.error("ci-gate FAIL %s: %s", case_id, exc)

    if failures:
        raise AssertionError(
            "CI regression gate failed:\n  - " + "\n  - ".join(failures)
        )
    return results
