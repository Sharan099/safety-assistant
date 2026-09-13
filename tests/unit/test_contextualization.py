"""Summary-augmented chunking: retrieval text construction, summary caching, failure fallback,
and the contamination boundary (the summary never reaches evidence)."""

from __future__ import annotations

import datetime
import uuid

from safety_assistant.contextualization import (
    SUMMARY_PROMPT_VERSION,
    build_retrieval_text,
    document_context,
    summary_cache_key,
)
from safety_assistant.contextualization.document_summary import MIN_SUMMARY_CHARS
from safety_assistant.evaluation.retrieval_eval import document_metrics
from safety_assistant.generation.prompts.grounded_v1 import format_evidence
from safety_assistant.persistence.models import Regulation, RegulationVersion
from safety_assistant.retrieval.context import Evidence, LegRanks
from safety_assistant.retrieval.sparse import tokenize

SUMMARY = "UN Regulation No. 94 concerns occupant protection in frontal collision tests of M1 vehicles."


def _reg() -> Regulation:
    return Regulation(
        regulation_key="UN-R94",
        title="Frontal collision protection",
        kind="REGULATION",
        authority="UNECE",
        jurisdiction="UNECE-1958-AGREEMENT",
        authority_level="AUTHORITATIVE",
        organization_id=uuid.uuid4(),
    )


def _version(**over: object) -> RegulationVersion:
    kw: dict[str, object] = dict(
        version_label="Rev.4 (04 series)", series="04", revision="Rev.4", valid_from=datetime.date(2021, 6, 9)
    )
    kw.update(over)
    return RegulationVersion(regulation_id=uuid.uuid4(), source_artifact_id=uuid.uuid4(), **kw)  # type: ignore[arg-type]


def test_document_context_is_deterministic_and_omits_unknowns() -> None:
    ctx = document_context(_reg(), _version())
    assert ctx.startswith("SOURCE DOCUMENT\nRegulation: UN R94\n")
    assert "Series: 04" in ctx and "Revision: Rev.4" in ctx and "Effective from: 2021-06-09" in ctx
    assert "Effective to" not in ctx  # NULL valid_to is omitted, never guessed
    sparse = document_context(_reg(), _version(series=None, revision=None, valid_from=None))
    assert "Series" not in sparse and "Effective" not in sparse
    assert document_context(_reg(), _version()) == ctx  # pure function of the rows


def test_retrieval_text_wraps_but_never_alters_chunk_content() -> None:
    content = "UN R94 › 5 Specifications › 5.2.1.8\n5.2.1.8. The thorax compression criterion shall not exceed 50 mm."
    ctx = document_context(_reg(), _version())
    with_summary = build_retrieval_text(ctx, SUMMARY, content)
    without = build_retrieval_text(ctx, None, content)
    assert with_summary.endswith(content) and without.endswith(content)
    assert "DOCUMENT SUMMARY" in with_summary and "DOCUMENT SUMMARY" not in without
    # the failure fallback (no summary) still carries document identity for both legs
    assert "Regulation: UN R94" in without
    assert {"04", "frontal", "collision"} <= set(tokenize(with_summary))
    assert "frontal" not in tokenize(content)  # the discriminating term only exists in the summary


def test_summary_cache_key_changes_with_content_prompt_or_model() -> None:
    base = summary_cache_key("sha-a", SUMMARY_PROMPT_VERSION, "model-x")
    assert base == summary_cache_key("sha-a", SUMMARY_PROMPT_VERSION, "model-x")
    assert base != summary_cache_key("sha-b", SUMMARY_PROMPT_VERSION, "model-x")
    assert base != summary_cache_key("sha-a", SUMMARY_PROMPT_VERSION + "x", "model-x")
    assert base != summary_cache_key("sha-a", SUMMARY_PROMPT_VERSION, "model-y")
    assert MIN_SUMMARY_CHARS < len(SUMMARY)


def _evidence(content: str) -> Evidence:
    return Evidence(
        evidence_id="E1",
        chunk_id=uuid.uuid4(),
        regulation_key="UN-R94",
        regulation_title="t",
        kind="REGULATION",
        jurisdiction="UNECE",
        authority_level="AUTHORITATIVE",
        version_id=uuid.uuid4(),
        version_label="Rev.4 (04 series)",
        version_status="ACTIVE",
        valid_from=None,
        valid_to=None,
        published_at=None,
        section_id=uuid.uuid4(),
        section_path="5.2.1.8",
        section_number="5.2.1.8.",
        section_title=None,
        annex=None,
        normative=True,
        chunk_type="TEXT",
        page_start=13,
        page_end=13,
        citation_label="UN R94 Rev.4 §5.2.1.8 (p. 13)",
        content=content,
        ranks=LegRanks(fused_score=1.0),
        source_sha256="x",
        source_uri=None,
        storage_uri="s",
        token_count=10,
    )


def test_summary_cannot_reach_the_generator_prompt() -> None:
    """The evidence model carries chunk content only; the prompt therefore cannot contain the
    summary even when the SAC index found the chunk. `Evidence` has no retrieval_text field."""
    assert "retrieval_text" not in Evidence.model_fields
    content = "5.2.1.8. The thorax compression criterion shall not exceed 50 mm."
    block = format_evidence([_evidence(content)])
    assert content in block and "DOCUMENT SUMMARY" not in block and "frontal" not in block


def test_document_metrics_and_drm_definition() -> None:
    exp = {"UN-R94"}
    m = document_metrics(["UN-R95", "UN-R94", "UN-R95"], exp, eligible=True)
    assert m["doc_recall@1"] == 0.0 and m["doc_recall@3"] == 1.0 and m["doc_mrr"] == 0.5
    assert m["drm@1"] == 1.0 and m["drm@5"] == 1.0  # a wrong document outranks the right one
    m = document_metrics(["UN-R94", "UN-R95"], exp, eligible=True)
    assert m["drm@1"] == 0.0 and m["drm@5"] == 0.0  # wrong document *below* the right one is not a mismatch
    m = document_metrics(["UN-R95"] * 6, exp, eligible=True)
    assert m["doc_recall@5"] == 0.0 and m["drm@5"] == 1.0
    m = document_metrics(["UN-R95", "UN-R94"], exp, eligible=False)
    assert m["drm@1"] is None and m["doc_recall@3"] == 1.0  # ineligible cases still get recall, never DRM
    assert document_metrics(["UN-R95"], set(), eligible=True)["doc_mrr"] is None


def test_summary_validator_rejects_reasoning_truncation_and_markdown() -> None:
    from safety_assistant.contextualization.document_summary import validate_summary

    good = SUMMARY + " It defines vehicle applicability, test conditions and injury assessment criteria."
    assert validate_summary(good, "stop") is None
    assert validate_summary(good, "length") == "truncated by max_tokens"
    assert "reasoning" in (
        validate_summary("The user wants me to create a short factual summary of " + good, "stop") or ""
    )
    assert "reasoning" in (validate_summary("Okay, let me analyze the document. " + good, "stop") or "")
    md = "**Document Type**: UN R129 **Scope**: child restraints **Requirements**: many **Tests**: dynamic " + good
    assert "markdown" in (validate_summary(md, "stop") or "")
    assert "length" in (validate_summary("Too short.", "stop") or "")


def test_compact_context_is_one_identity_line_plus_first_summary_sentence() -> None:
    from safety_assistant.contextualization.context_builder import build_compact_text, compact_context

    line = compact_context(_reg(), _version(), SUMMARY + " Second sentence about annexes.")
    assert line.startswith("UN R94 — Frontal collision protection (Rev.4 (04 series)).")
    assert line.endswith(SUMMARY) and "Second sentence" not in line
    assert "\n" not in line
    assert compact_context(_reg(), _version(), None) == "UN R94 — Frontal collision protection (Rev.4 (04 series))."
    text = build_compact_text(line, "5.2.1.8. The thorax compression criterion shall not exceed 42 mm.")
    assert text.split("\n", 1)[1].startswith("5.2.1.8.")  # content immediately after the one line
