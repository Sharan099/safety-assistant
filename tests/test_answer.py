"""Unit tests for answer assembly (no Qdrant / embedder required)."""

from __future__ import annotations

import json
from pathlib import Path

from generation.answer import (
    answer_question,
    assert_prose_sections_match_citations,
    log_groundedness_violation,
    parse_structured_answer,
    render_answer_from_segments,
    validate_segment_chunk_ids,
)
from generation.llm_client import LLMClient
from retrieval.retrieve import RetrievedChunk, format_context


def _chunk(**kwargs) -> RetrievedChunk:
    base = dict(
        chunk_id="c1",
        text="The HIC shall not exceed 1000.",
        regulation_id="UN-ECE-R94",
        revision="Rev.3",
        section_number="5.2.1",
        section_title="Head Injury Criterion",
        page_number=12,
        bounding_box=[10.0, 20.0, 100.0, 40.0],
        content_type="clause",
        section_id="UN-ECE-R94::5.2.1",
        score=0.91,
    )
    base.update(kwargs)
    return RetrievedChunk(**base)


def test_citation_tag():
    c = _chunk()
    assert c.citation_tag() == "[UN-ECE-R94 §5.2.1, p.12]"


def test_format_context_includes_chunk_id():
    ctx = format_context([_chunk()])
    assert "chunk_id=c1" in ctx
    assert "HIC shall not exceed 1000" in ctx
    assert "citation_chunk_id=c1" in ctx


def test_answer_with_injected_chunks(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "generation.answer.GROUNDEDNESS_VIOLATIONS_PATH",
        tmp_path / "groundedness_violations.jsonl",
    )
    llm = LLMClient(
        provider="mock",
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "llm.jsonl",
    )
    result = answer_question(
        "What is the HIC limit in R94?",
        chunks=[_chunk()],
        llm=llm,
        skip_answer_cache=True,
    )
    assert result.answer
    assert "[UN-ECE-R94 §5.2.1, p.12]" in result.answer
    assert len(result.sources) == 1
    src = result.sources[0]
    assert src.page_number == 12
    assert src.bounding_box == [10.0, 20.0, 100.0, 40.0]
    assert src.citation == "[UN-ECE-R94 §5.2.1, p.12]"
    assert src.chunk_id == "c1"
    assert result.provider == "mock"
    assert_prose_sections_match_citations(result.answer, result.sources)
    data = result.model_dump()
    assert "answer" in data and "sources" in data


def test_answer_empty_retrieval(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "generation.answer.retrieval_miss_message",
        lambda indexed=None: (
            "I couldn't find relevant content on this in the indexed regulations (none). "
            "Try rephrasing, specifying a clause number, or upload the relevant regulation "
            "if it isn't listed."
        ),
    )
    llm = LLMClient(
        provider="mock",
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "llm.jsonl",
    )
    result = answer_question("Anything?", chunks=[], llm=llm, skip_answer_cache=True)
    assert "couldn't find relevant content" in result.answer
    assert result.sources == []
    assert result.not_found is True
    assert result.failure_kind == "retrieval_miss"


def test_rejects_unknown_chunk_id(tmp_path: Path, monkeypatch):
    violations = tmp_path / "groundedness_violations.jsonl"
    monkeypatch.setattr("generation.answer.GROUNDEDNESS_VIOLATIONS_PATH", violations)

    calls = {"n": 0}

    def fake_complete(self, *args, **kwargs):
        calls["n"] += 1
        from generation.llm_client import LLMResult

        return LLMResult(
            text=json.dumps(
                {
                    "answer_segments": [
                        {"text": "Invented claim", "citation_chunk_id": "not-a-real-id"}
                    ]
                }
            ),
            model="mock-model",
            provider="mock",
            role="answer",
            input_tokens=1,
            output_tokens=1,
        )

    monkeypatch.setattr(LLMClient, "complete", fake_complete)
    llm = LLMClient(
        provider="mock",
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "llm.jsonl",
        use_cache=False,
    )
    result = answer_question(
        "What is HIC?",
        chunks=[_chunk()],
        llm=llm,
        skip_answer_cache=True,
    )
    assert result.failure_kind == "grounding_rejected"
    assert result.not_found is True
    assert "couldn't produce a confidently grounded answer" in result.answer
    assert "[UN-ECE-R94 §5.2.1, p.12]" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "c1"
    assert calls["n"] == 2  # initial + retry
    assert violations.is_file()
    lines = violations.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    assert rec["question"] == "What is HIC?"
    assert "c1" in rec["retrieved_chunk_ids"]
    assert "not-a-real-id" in rec["invalid_citation_chunk_ids"]


def test_render_uses_metadata_not_model_section():
    structured = parse_structured_answer(
        json.dumps(
            {
                "answer_segments": [
                    {
                        "text": "HIC shall not exceed 1000.",
                        "citation_chunk_id": "c1",
                    }
                ]
            }
        )
    )
    assert validate_segment_chunk_ids(structured, {"c1"}) == []
    answer, sources = render_answer_from_segments(structured, {"c1": _chunk()})
    assert sources[0].section_number == "5.2.1"
    assert "[UN-ECE-R94 §5.2.1, p.12]" in answer
    assert_prose_sections_match_citations(answer, sources)


def test_assert_prose_mismatch_fails():
    from generation.answer import SourceChunk

    sources = [
        SourceChunk(
            chunk_id="c1",
            regulation_id="UN-ECE-R94",
            section_number="5.2.1",
            page_number=12,
            citation="[UN-ECE-R94 §5.2.1, p.12]",
        )
    ]
    # Chip claims a different section than the source metadata.
    bad = "Limit is 1000 [UN-ECE-R94 §9.9.9, p.12]"
    try:
        assert_prose_sections_match_citations(bad, sources)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_log_violation_appends(tmp_path: Path):
    path = tmp_path / "groundedness_violations.jsonl"
    log_groundedness_violation(
        question="Define H-point",
        model_output='{"answer_segments":[]}',
        retrieved_chunk_ids=["efaa3fe01ca6c6fd"],
        invalid_ids=["bogus"],
        path=path,
    )
    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["question"] == "Define H-point"
    assert rec["invalid_citation_chunk_ids"] == ["bogus"]
