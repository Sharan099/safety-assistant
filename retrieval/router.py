"""Query intent router — classify once, drive distinct retrieval/generation pipelines.

Regex fast-paths first (deterministic); optional cheap LLM classifier when ambiguous.
Every classification is appended to ``eval/results/query_intent_audit.jsonl``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_PATH = ROOT / "eval" / "results" / "query_intent_audit.jsonl"


class QueryIntent(str, Enum):
    FACTUAL_LOOKUP = "FACTUAL_LOOKUP"
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    DESIGN_IMPLICATION = "DESIGN_IMPLICATION"
    CHECKLIST_GEN = "CHECKLIST_GEN"
    SCOPE_SUMMARY = "SCOPE_SUMMARY"
    APPLICABILITY = "APPLICABILITY"
    RETEST_SCOPE = "RETEST_SCOPE"


class RetrievalStrategy(str, Enum):
    """How hybrid/retrieve should behave for this intent."""

    STANDARD = "standard"  # top-k factual
    COMPLIANCE = "compliance"  # limits + optional retrieve for citations
    MULTI_REG_TOPIC = "multi_reg_topic"  # per indexed reg, topic-filtered
    CHECKLIST = "checklist"  # enumerative breadth + template subqueries
    SCOPE_ARTICLE = "scope_article"  # hard-reg + scope prepend + broad recall
    APPLICABILITY_SCOPES = "applicability_scopes"  # scope clauses across regs
    RETEST_IMPACT = "retest_impact"  # applicability + change-impact recall


class GroundingRule(str, Enum):
    STRICT_CITATION = "strict_citation"  # every claim cites a chunk_id
    DETERMINISTIC_VERDICT = "deterministic_verdict"  # Fix 22 structured pass/fail
    MULTI_REG_COVERAGE = "multi_reg_coverage"  # address each reg; state gaps
    CHECKLIST_ITEMS = "checklist_items"  # one cited item per requirement found
    STRUCTURED_SUMMARY = "structured_summary"  # scope/applicability sections
    CHANGE_IMPACT = "change_impact"  # what must be retested; abstain if silent


@dataclass(frozen=True)
class PipelineConfig:
    """Per-intent retrieval / budget / grounding — not a shared global config."""

    intent: QueryIntent
    retrieval_strategy: RetrievalStrategy
    max_chunks: int
    max_tokens: int
    hybrid_top_k: int
    rerank_top_k: int
    hard_reg_filter: bool  # apply named-reg Qdrant filter when a single reg is named
    multi_reg_loop: bool
    prepend_scope: bool
    small_to_big: bool
    grounding_rule: GroundingRule
    budget_mode: str
    prompt_instruction: str = ""


# Distinct configs — do not collapse these into one shared table of defaults.
PIPELINE_CONFIGS: dict[QueryIntent, PipelineConfig] = {
    QueryIntent.FACTUAL_LOOKUP: PipelineConfig(
        intent=QueryIntent.FACTUAL_LOOKUP,
        retrieval_strategy=RetrievalStrategy.STANDARD,
        max_chunks=5,
        max_tokens=3000,
        hybrid_top_k=30,
        rerank_top_k=5,
        hard_reg_filter=True,
        multi_reg_loop=False,
        prepend_scope=False,
        small_to_big=True,
        grounding_rule=GroundingRule.STRICT_CITATION,
        budget_mode="factual_lookup",
        prompt_instruction=(
            "FACTUAL LOOKUP — answer from the cited clause only; prefer a single "
            "precise claim with one citation_chunk_id."
        ),
    ),
    QueryIntent.COMPLIANCE_CHECK: PipelineConfig(
        intent=QueryIntent.COMPLIANCE_CHECK,
        retrieval_strategy=RetrievalStrategy.COMPLIANCE,
        max_chunks=5,
        max_tokens=3000,
        hybrid_top_k=30,
        rerank_top_k=5,
        hard_reg_filter=True,
        multi_reg_loop=False,
        prepend_scope=False,
        small_to_big=False,
        grounding_rule=GroundingRule.DETERMINISTIC_VERDICT,
        budget_mode="compliance_check",
        prompt_instruction=(
            "COMPLIANCE CHECK — emit an explicit PASS/FAIL/CANNOT_DETERMINE verdict; "
            "do not stop at a procedure description."
        ),
    ),
    QueryIntent.DESIGN_IMPLICATION: PipelineConfig(
        intent=QueryIntent.DESIGN_IMPLICATION,
        retrieval_strategy=RetrievalStrategy.MULTI_REG_TOPIC,
        # Fix 13: multi-retrieval must stay under a hard cap (was 16/8k → 57s/22k).
        max_chunks=10,
        max_tokens=4500,
        hybrid_top_k=30,
        rerank_top_k=8,
        hard_reg_filter=False,
        multi_reg_loop=True,
        prepend_scope=False,
        small_to_big=False,
        grounding_rule=GroundingRule.MULTI_REG_COVERAGE,
        budget_mode="design_implication",
        prompt_instruction=(
            "DESIGN IMPLICATION — list requirements that affect the named component "
            "or design decision across indexed regulations; each claim cites its own "
            "chunk_id (per-claim grounding). Label REGULATORY_FACT vs "
            "ENGINEERING_INFERENCE explicitly; never present engineering judgment as "
            "cited regulation text. State which indexed regulations have no relevant content."
        ),
    ),
    QueryIntent.CHECKLIST_GEN: PipelineConfig(
        intent=QueryIntent.CHECKLIST_GEN,
        retrieval_strategy=RetrievalStrategy.CHECKLIST,
        max_chunks=14,
        max_tokens=5500,
        hybrid_top_k=30,
        rerank_top_k=14,
        hard_reg_filter=True,
        multi_reg_loop=False,
        prepend_scope=False,
        small_to_big=False,
        grounding_rule=GroundingRule.CHECKLIST_ITEMS,
        budget_mode="checklist_gen",
        prompt_instruction=(
            "CHECKLIST GENERATION — produce a preparation/homologation checklist "
            "grouped by category (vehicle prep, dummy installation, instrumentation, "
            "injury criteria, documentation, …); each item cites a clause; explicitly "
            "note categories with no indexed content (silence ≠ not required)."
        ),
    ),
    QueryIntent.SCOPE_SUMMARY: PipelineConfig(
        intent=QueryIntent.SCOPE_SUMMARY,
        retrieval_strategy=RetrievalStrategy.SCOPE_ARTICLE,
        max_chunks=10,
        max_tokens=4500,
        hybrid_top_k=30,
        rerank_top_k=8,
        hard_reg_filter=True,
        multi_reg_loop=False,
        prepend_scope=True,
        small_to_big=False,
        grounding_rule=GroundingRule.STRUCTURED_SUMMARY,
        budget_mode="scope_summary",
        prompt_instruction=(
            "SCOPE SUMMARY — summarize ONE named regulation only (hard-filtered). "
            "Structure as Scope/applicability, Key injury criteria (verified limits), "
            "Test configuration, Homologation obligations — each cited. Never cite "
            "another regulation."
        ),
    ),
    QueryIntent.APPLICABILITY: PipelineConfig(
        intent=QueryIntent.APPLICABILITY,
        retrieval_strategy=RetrievalStrategy.APPLICABILITY_SCOPES,
        max_chunks=12,
        max_tokens=5000,
        hybrid_top_k=30,
        rerank_top_k=8,
        hard_reg_filter=False,
        multi_reg_loop=True,
        prepend_scope=True,
        small_to_big=False,
        grounding_rule=GroundingRule.STRUCTURED_SUMMARY,
        budget_mode="applicability",
        prompt_instruction=(
            "APPLICABILITY — survey EVERY indexed regulation's Scope clause for the "
            "described vehicle; emit APPLIES / DOES_NOT_APPLY / CANNOT_DETERMINE per "
            "regulation with a cited scope passage. Never answer with only one "
            "regulation when multiple are indexed (e.g. passenger EV → R94 and R95)."
        ),
    ),
    QueryIntent.RETEST_SCOPE: PipelineConfig(
        intent=QueryIntent.RETEST_SCOPE,
        retrieval_strategy=RetrievalStrategy.RETEST_IMPACT,
        max_chunks=10,
        max_tokens=4500,
        hybrid_top_k=30,
        rerank_top_k=8,
        hard_reg_filter=True,
        multi_reg_loop=False,
        prepend_scope=False,
        small_to_big=False,
        grounding_rule=GroundingRule.CHANGE_IMPACT,
        budget_mode="retest_scope",
        prompt_instruction=(
            "RETEST SCOPE — cite modification / extension-of-approval clauses; "
            "relate the change as analysis only. NEVER issue an authoritative "
            "retest decision — frame as governing-clause information for the "
            "homologation authority. If the indexed text is silent, say so."
        ),
    ),
}


# --- Regex fast-paths (priority order in classify_regex) --------------------

_COMPLIANCE_RE = re.compile(
    r"(?ix)\b("
    r"pass(?:es|ed)?|fail(?:s|ed)?|comply|complies|compliance|"
    r"satisf(?:y|ies|ied)|conform(?:s|ity)?|"
    r"does\s+(?:the\s+)?(?:vehicle|it)\s+(?:pass|comply)|"
    r"within\s+(?:the\s+)?limit|exceed(?:s|ed)?"
    r")\b"
)

_CHECKLIST_RE = re.compile(
    r"(?ix)\b("
    r"checklist|"
    r"prepare(?:ing)?\s+a\s+(?:vehicle|test)|"
    r"homologat(?:e|ion|ing)|"
    r"generate\s+a\s+checklist|"
    r"preparation\s+(?:checklist|steps)|"
    r"steps\s+to\s+prepare"
    r")\b"
)

_RETEST_RE = re.compile(
    r"(?ix)\b("
    r"re-?test|"
    r"re-?testing|"
    r"after\s+(?:chang(?:e|ing)|modif(?:y|ying|ication)|replacing|updating)\b|"
    r"need\s+to\s+re-?test|"
    r"does\s+(?:this|the)\s+change\s+(?:require|affect|trigger)|"
    r"impact\s+on\s+(?:approval|type\s+approval|testing)|"
    r"affected\s+tests?"
    r")"
)

_DESIGN_RE = re.compile(
    r"(?ix)\b("
    r"design\s+implication|"
    r"what\s+requirements?\b.{0,80}?\b(?:affect|apply\s+to|constrain)\b|"
    r"requirements?\b.{0,60}?\baffect\b|"
    r"how\s+(?:does|do|should)\s+(?:this|the)\s+(?:design|component|decision)|"
    r"engineering\s+(?:impact|implication)|"
    r"implications?\s+for\s+(?:the\s+)?(?:design|component|structure|door|seat|belt|reess)|"
    r"which\s+requirements?\s+(?:must|should)\s+(?:we|I)\s+consider\s+for|"
    r"affect\s+(?:our\s+)?(?:vehicle\s+)?design\b"
    r")"
)

_SCOPE_SUMMARY_RE = re.compile(
    r"(?ix)\b("
    r"summar(?:y|ize|ise)\s+(?:the\s+)?(?:scope|requirements?|regulation)|"
    r"overview\s+of\s+(?:UN\s*)?R?\d+|"
    r"what\s+(?:is|are)\s+(?:the\s+)?(?:scope|purpose|objective)\s+of|"
    r"what\s+does\s+(?:UN\s*)?R?\d+\s+(?:cover|address|require)\b|"
    r"whole[- ]regulation\s+summary|"
    r"key\s+requirements?\s+of\s+(?:UN\s*)?R?\d+"
    r")"
)

# Corpus vehicle/use-case applicability only (Fix 24/25).
# Do NOT match "vehicles covered under UN R94" (named-reg scope) or
# "which regulations include/cover <topic>" (Fix 25 topic survey).
_APPLICABILITY_RE = re.compile(
    r"(?ix)\b("
    r"which\s+regulations?\s+(?:apply|are\s+applicable)\b|"
    r"what\s+regulations?\s+(?:apply|are\s+applicable)\b|"
    r"which\s+regulations?\s+are\s+applicable\b|"
    r"applicable\s+(?:to|for)\b|"
    r"are\s+applicable\s+(?:to|for|because)\b|"
    r"\bapplicability\b|"
    r"(?:does|do)\s+(?:UN\s*)?R?\d+\s+apply\s+to|"
    r"for\s+(?:an?\s+)?(?:M1|N1|M2|N2|electric|hybrid|ICE)\b.+\b(?:which|what)\s+reg"
    r")"
)

_LLM_INTENT_NAMES = {i.value for i in QueryIntent}

CLASSIFIER_SYSTEM = """\
You classify UNECE passive-safety engineer questions into exactly one intent.
Return ONLY valid JSON: {"intent":"<NAME>","reason":"<short>"}.

Intents:
- FACTUAL_LOOKUP: single fact / limit / definition / clause citation
- COMPLIANCE_CHECK: measured values or qualitative facts vs pass/fail / comply
- DESIGN_IMPLICATION: which requirements affect a component or design decision
- CHECKLIST_GEN: generate a checklist for preparing / homologating / testing
- SCOPE_SUMMARY: summarize scope or overall requirements of one named regulation
- APPLICABILITY: which regulations apply to a vehicle/use-case (no single named reg)
- RETEST_SCOPE: after a change, what must be retested / re-approved

Prefer COMPLIANCE_CHECK over FACTUAL_LOOKUP when pass/fail/comply is asked.
Prefer CHECKLIST_GEN over SCOPE_SUMMARY when a checklist/preparation list is asked.
Prefer RETEST_SCOPE when the user describes a modification and asks about retesting.
Prefer FACTUAL_LOOKUP when a single regulation is named (e.g. "covered under UN R94").
Prefer FACTUAL_LOOKUP for topic surveys ("which regulations include electrical…") —
  those are multi-regulation retrieval, not vehicle APPLICABILITY.
"""


@dataclass
class RoutedQuery:
    intent: QueryIntent
    reason: str
    source: str  # regex | llm | default
    regulation_id: str | None = None
    confidence: float = 1.0
    pipeline: PipelineConfig = field(
        default_factory=lambda: PIPELINE_CONFIGS[QueryIntent.FACTUAL_LOOKUP]
    )
    flags: dict[str, Any] = field(default_factory=dict)
    question: str = ""
    condensed: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "reason": self.reason,
            "source": self.source,
            "regulation_id": self.regulation_id,
            "confidence": self.confidence,
            "budget_mode": self.pipeline.budget_mode,
            "retrieval_strategy": self.pipeline.retrieval_strategy.value,
            "grounding_rule": self.pipeline.grounding_rule.value,
            "max_chunks": self.pipeline.max_chunks,
            "max_tokens": self.pipeline.max_tokens,
            "flags": dict(self.flags),
        }


def pipeline_for(intent: QueryIntent) -> PipelineConfig:
    return PIPELINE_CONFIGS[intent]


def classify_regex(question: str) -> tuple[QueryIntent, str] | None:
    """Deterministic fast-path. Returns (intent, reason) or None if ambiguous."""
    q = (question or "").strip()
    if not q:
        return QueryIntent.FACTUAL_LOOKUP, "empty→factual"

    named: str | None = None
    try:
        from retrieval.enumerative import detect_named_regulation

        named = detect_named_regulation(q)
    except Exception:  # noqa: BLE001
        named = None

    # Fix 24: "What vehicles are covered under UN R94?" is single-reg scope —
    # never corpus APPLICABILITY (that would survey all regs without a hard filter).
    if named and re.search(
        r"(?ix)\b(?:what|which)\s+vehicles?\b|\bvehicles?\s+(?:are\s+)?covered\b|"
        r"\bcovered\s+under\b",
        q,
    ):
        return QueryIntent.FACTUAL_LOOKUP, "regex:named_reg_coverage→factual"

    # Priority: retest > compliance > checklist > design > applicability > scope > None
    if _RETEST_RE.search(q):
        return QueryIntent.RETEST_SCOPE, f"regex:retest"
    if _COMPLIANCE_RE.search(q):
        try:
            from generation.compliance import is_compliance_check_query

            if is_compliance_check_query(q) or re.search(
                r"(?i)\bdoes\s+(?:the\s+)?(?:vehicle|it)\b|\bpass\b|\bfail\b|\bcomply\b",
                q,
            ):
                return QueryIntent.COMPLIANCE_CHECK, "regex:compliance"
        except Exception:  # noqa: BLE001
            return QueryIntent.COMPLIANCE_CHECK, "regex:compliance_cue"
    if _CHECKLIST_RE.search(q):
        return QueryIntent.CHECKLIST_GEN, "regex:checklist"
    if _DESIGN_RE.search(q):
        return QueryIntent.DESIGN_IMPLICATION, "regex:design"
    if _APPLICABILITY_RE.search(q):
        # Fix 24: a single named regulation is never a corpus-wide applicability survey.
        if named:
            return QueryIntent.FACTUAL_LOOKUP, "regex:named_reg_not_applicability"
        return QueryIntent.APPLICABILITY, "regex:applicability"
    if _SCOPE_SUMMARY_RE.search(q):
        return QueryIntent.SCOPE_SUMMARY, "regex:scope_summary"

    # Strong factual cues — accept without LLM.
    if re.search(
        r"(?ix)\b("
        r"what\s+is\s+(?:the\s+)?(?:limit|definition|value|threshold|HPC|ThCC|RDC|VC)|"
        r"define\b|definition\s+of|"
        r"shall\s+not\s+exceed|"
        r"clause\s+\d|"
        r"§\s*\d"
        r")\b",
        q,
    ):
        return QueryIntent.FACTUAL_LOOKUP, "regex:factual_cue"

    return None


def _llm_classify(question: str, *, llm: Any | None) -> tuple[QueryIntent, str, float] | None:
    if llm is None:
        return None
    if (os.getenv("QUERY_ROUTER_LLM") or "1").strip().lower() in {"0", "false", "off", "no"}:
        return None
    try:
        from generation.llm_client import LLMClient, LLMRole

        client = llm if isinstance(llm, LLMClient) else LLMClient()
        if getattr(client, "provider", "mock") == "mock":
            return None
        result = client.complete(
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user", "content": question},
            ],
            role=LLMRole.REWRITE,
            question=f"intent:{question[:80]}",
            temperature=0.0,  # Fix 12 — classifier is non-creative
            seed=42,
            max_tokens=80,
            skip_cache=False,
        )
        text = (result.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        name = str(data.get("intent") or "").strip().upper()
        if name not in _LLM_INTENT_NAMES:
            return None
        reason = str(data.get("reason") or "llm").strip()[:200]
        return QueryIntent(name), f"llm:{reason}", 0.7
    except Exception as exc:  # noqa: BLE001
        logger.warning("query router LLM classify failed: %s", exc)
        return None


def _compose_flags(question: str, intent: QueryIntent) -> dict[str, Any]:
    flags: dict[str, Any] = {"intent": intent.value}
    try:
        from retrieval.enumerative import is_enumerative_query, detect_named_regulation
        from retrieval.multi_regulation import is_plural_regulation_query
        from retrieval.multi_criterion import is_multi_criterion_query
        from retrieval.value_limit import is_value_vs_limit_query, is_named_criterion_query

        flags["enumerative"] = is_enumerative_query(question)
        flags["plural_regulation"] = is_plural_regulation_query(question)
        flags["multi_criterion"] = is_multi_criterion_query(question)
        flags["value_vs_limit"] = is_value_vs_limit_query(question)
        flags["named_criterion"] = is_named_criterion_query(question)
        flags["named_regulation_id"] = detect_named_regulation(question)
    except Exception as exc:  # noqa: BLE001
        flags["flag_error"] = str(exc)[:120]
    return flags


def log_classification(
    routed: RoutedQuery,
    *,
    path: Path | None = None,
) -> None:
    """Append one audit record for misroute review."""
    out = path or Path(
        (os.getenv("QUERY_INTENT_AUDIT_PATH") or str(DEFAULT_AUDIT_PATH)).strip()
        or str(DEFAULT_AUDIT_PATH)
    )
    rec = {
        "ts": time.time(),
        "question": routed.question,
        "condensed": routed.condensed,
        **routed.to_public_dict(),
    }
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("query intent audit log failed: %s", exc)
    logger.info(
        "query_intent intent=%s source=%s reason=%r strategy=%s budget=%s reg=%s q=%r",
        routed.intent.value,
        routed.source,
        routed.reason,
        routed.pipeline.retrieval_strategy.value,
        routed.pipeline.budget_mode,
        routed.regulation_id,
        (routed.question or "")[:120],
    )


def classify_query(
    question: str,
    *,
    condensed: str | None = None,
    llm: Any | None = None,
    use_llm: bool | None = None,
    log: bool = True,
) -> RoutedQuery:
    """Classify ``question`` (prefer original text for cues; ``condensed`` for flags).

    Order: regex fast-path → optional cheap LLM → default FACTUAL_LOOKUP.
    """
    original = (question or "").strip()
    stand = (condensed or original).strip() or original
    probe = original or stand

    named_reg: str | None = None
    try:
        from retrieval.enumerative import detect_named_regulation, resolve_hard_regulation_filter

        named_reg = resolve_hard_regulation_filter(probe) or detect_named_regulation(probe)
    except Exception:  # noqa: BLE001
        named_reg = None

    hit = classify_regex(probe)
    if hit is None and stand != probe:
        hit = classify_regex(stand)

    source = "regex"
    confidence = 1.0
    if hit is not None:
        intent, reason = hit
    else:
        llm_hit = None
        if use_llm is None:
            use_llm = (os.getenv("QUERY_ROUTER_LLM") or "1").strip().lower() not in {
                "0",
                "false",
                "off",
                "no",
            }
        if use_llm:
            llm_hit = _llm_classify(probe, llm=llm)
        if llm_hit is not None:
            intent, reason, confidence = llm_hit
            source = "llm"
        else:
            intent, reason, source, confidence = (
                QueryIntent.FACTUAL_LOOKUP,
                "default:no_regex_no_llm",
                "default",
                0.4,
            )

    # Compatibility: enumerative list → CHECKLIST_GEN
    if intent == QueryIntent.FACTUAL_LOOKUP:
        try:
            from retrieval.enumerative import is_enumerative_query

            if is_enumerative_query(probe) and re.search(
                r"(?i)\blist\b|\bevery\b|\ball\s+the\s+requirements\b", probe
            ):
                intent = QueryIntent.CHECKLIST_GEN
                reason = "regex:enumerative_list→checklist"
                source = "regex"
                confidence = 0.9
        except Exception:  # noqa: BLE001
            pass

    pipe = pipeline_for(intent)
    flags = _compose_flags(stand or probe, intent)
    routed = RoutedQuery(
        intent=intent,
        reason=reason,
        source=source,
        regulation_id=named_reg,
        confidence=confidence,
        pipeline=pipe,
        flags=flags,
        question=original,
        condensed=stand,
    )
    if log:
        log_classification(routed)
    return routed


def budgets_for_intent(intent: QueryIntent | RoutedQuery) -> tuple[int, int, str]:
    """Return ``(max_chunks, max_tokens, mode)`` from the intent's pipeline config.

    Fix 13: factual/compliance stay under the standard hard ceiling (5 / 3000);
    Layer 4–5 intents keep their wider pipeline budgets.
    """
    if isinstance(intent, RoutedQuery):
        pipe = intent.pipeline
    else:
        pipe = pipeline_for(intent)
    env_chunks = os.getenv(f"INTENT_{pipe.intent.value}_MAX_CHUNKS")
    env_tokens = os.getenv(f"INTENT_{pipe.intent.value}_MAX_TOKENS")
    chunks = int(env_chunks) if env_chunks else pipe.max_chunks
    tokens = int(env_tokens) if env_tokens else pipe.max_tokens
    chunks, tokens = max(1, chunks), max(500, tokens)
    from retrieval.context_budget import STANDARD_BUDGET_MODES, STANDARD_MAX_CHUNKS, STANDARD_MAX_TOKENS

    if pipe.budget_mode in STANDARD_BUDGET_MODES:
        chunks = min(chunks, STANDARD_MAX_CHUNKS)
        tokens = min(tokens, STANDARD_MAX_TOKENS)
    return chunks, tokens, pipe.budget_mode
