"""Exclusive failure-state taxonomy: retrieval_miss vs grounding_rejected."""

from __future__ import annotations

import json
from pathlib import Path

from generation.answer import (
    FAILURE_GROUNDING_REJECTED,
    FAILURE_RETRIEVAL_MISS,
    answer_question,
    grounding_rejected_message,
    retrieval_miss_message,
)
from generation.llm_client import LLMClient, LLMResult
from retrieval.retrieve import IndexedRegulation, RetrievedChunk


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


def test_messages_are_distinct():
    miss = retrieval_miss_message(
        indexed=[IndexedRegulation(regulation_id="UN-ECE-R94", revision="x", chunk_count=1)]
    )
    rejected = grounding_rejected_message(_chunk())
    assert "couldn't find relevant content" in miss
    assert "confidently grounded" in rejected
    assert miss != rejected
    assert "§5.2.1" in rejected


def test_empty_llm_segments_with_chunks_is_grounding_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "generation.answer.GROUNDEDNESS_VIOLATIONS_PATH",
        tmp_path / "violations.jsonl",
    )

    def fake_complete(self, *args, **kwargs):
        return LLMResult(
            text=json.dumps({"answer_segments": []}),
            model="mock",
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
        "How does R16 relate to R94?",
        chunks=[_chunk()],
        llm=llm,
        skip_answer_cache=True,
    )
    assert result.failure_kind == FAILURE_GROUNDING_REJECTED
    assert result.not_found is True
    assert "confidently grounded" in result.answer
    assert "couldn't find relevant content" not in result.answer
    assert len(result.sources) == 1


def test_no_chunks_is_retrieval_miss_only(tmp_path: Path):
    llm = LLMClient(
        provider="mock",
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "llm.jsonl",
    )
    result = answer_question("Anything?", chunks=[], llm=llm, skip_answer_cache=True)
    assert result.failure_kind == FAILURE_RETRIEVAL_MISS
    assert result.not_found is True
    assert "couldn't find relevant content" in result.answer
    assert "confidently grounded" not in result.answer
    assert result.sources == []
