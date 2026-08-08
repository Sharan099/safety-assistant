"""Hard regulation_id filter for single-named-reg queries."""

from __future__ import annotations

from retrieval.enumerative import (
    detect_named_regulation,
    is_comparative_query,
    resolve_hard_regulation_filter,
)


def test_detect_named_single_and_multi():
    assert detect_named_regulation("What vehicles are covered under UN R94?") == "UN-ECE-R94"
    assert detect_named_regulation("minimum isolation resistance in UN R94") == "UN-ECE-R94"
    assert detect_named_regulation("HPC limit in R16") == "UN-ECE-R16"
    assert detect_named_regulation("How does UN R16 relate to UN R94?") is None
    assert detect_named_regulation("compare R94 and R95") is None


def test_comparative_skips_hard_filter():
    assert is_comparative_query("compare R94 and R95 frontal requirements")
    assert is_comparative_query("How does UN R16 relate to UN R94?")
    assert is_comparative_query("What is the difference between R94 and R95?")
    assert not is_comparative_query("What vehicles are covered under UN R94?")
    assert not is_comparative_query("What is the minimum isolation resistance requirement in UN R94?")

    assert resolve_hard_regulation_filter("What vehicles are covered under UN R94?") == "UN-ECE-R94"
    assert (
        resolve_hard_regulation_filter(
            "What is the minimum isolation resistance requirement in UN R94?"
        )
        == "UN-ECE-R94"
    )
    assert resolve_hard_regulation_filter("compare R94 and R95") is None
    assert resolve_hard_regulation_filter("How does UN R16 relate to UN R94?") is None
    # Comparative with a single named reg still skips (cross-reg intent).
    assert resolve_hard_regulation_filter("compare R94 with seat-belt rules") is None


def test_retrieve_vehicles_r94_never_r16():
    from retrieval.retrieve import retrieve

    q = "What vehicles are covered under UN R94?"
    chunks = retrieve(q, top_k=5, rewrite=False, do_rerank=True, small_to_big=False)
    assert chunks, "expected retrieval hits"
    regs = {c.regulation_id for c in chunks}
    assert regs == {"UN-ECE-R94"}, regs
    assert "UN-ECE-R16" not in regs
    secs = {(c.section_number or "").strip() for c in chunks}
    assert "1" in secs or any(s.startswith("1") for s in secs)


def test_retrieve_isolation_r94_annex_consistent():
    from retrieval.retrieve import retrieve

    q = "What is the minimum isolation resistance requirement in UN R94?"
    gold_ids = {
        "1faa4e17b012c1ab",
        "9a19875b521d408a",
        "1be49061bf4cd8e8",
        "86bfaf9c324cbf3f",
        "35b5fc1424f11bb3",
    }
    for _ in range(3):
        chunks = retrieve(q, top_k=5, rewrite=False, do_rerank=True, small_to_big=False)
        assert chunks
        regs = {c.regulation_id for c in chunks}
        assert regs == {"UN-ECE-R94"}, regs
        assert "UN-ECE-R16" not in regs
        joined = " ".join(
            f"{(c.text or '')} {(c.section_number or '')}" for c in chunks
        ).lower()
        assert "isolation" in joined
        hit_gold = any(c.chunk_id in gold_ids for c in chunks)
        annex = any("annex 11" in (c.section_number or "").lower() for c in chunks)
        assert hit_gold or annex or "annex 11" in joined
