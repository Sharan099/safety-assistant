"""Unit tests for eval.run_full (no live LLM / Qdrant)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from eval.run_full import (
    aggregate_per_category,
    append_partial_result,
    apply_thresholds,
    build_cost_summary,
    check_budget_ceiling,
    chunked,
    estimate_run_llm_calls,
    load_partial_results,
    load_run_config,
    load_thresholds,
    print_run_cost_summary,
    summarize_usage_since,
    write_run_config,
)


def test_load_thresholds_has_all_categories():
    thr = load_thresholds()
    cats = thr["categories"]
    for key in (
        "factual_lookup",
        "numeric_safety",
        "prompt_injection",
        "guardrail",
        "hallucination_probe",
        "out_of_scope",
    ):
        assert key in cats
        assert "severity" in cats[key]
    assert thr["call_budget"]["max_estimated_calls"] > 0


def test_estimate_counts_system_judge_guard():
    cases = [
        {"category": "factual_lookup"},
        {"category": "factual_lookup"},
        {"category": "guardrail"},
        {"category": "prompt_injection"},
        {"category": "numeric_safety"},
    ]
    est = estimate_run_llm_calls(
        cases, rewrite_enabled=True, skip_ragas=False, include_security_fp=True
    )
    assert est["system_calls"] > 0
    assert est["judge_calls"] > 0
    assert est["guard_calls"] > 0
    assert est["total"] == est["system_calls"] + est["judge_calls"] + est["guard_calls"]
    assert est["false_positive_probe"] is not None
    assert "factual_lookup" in est["by_category"]


def test_estimate_skip_ragas_zeros_judge_for_ragas_cats():
    cases = [{"category": "factual_lookup"}]
    with_j = estimate_run_llm_calls(cases, skip_ragas=False, include_security_fp=False)
    without = estimate_run_llm_calls(cases, skip_ragas=True, include_security_fp=False)
    assert without["judge_calls"] == 0
    assert with_j["judge_calls"] > without["judge_calls"]


def test_aggregate_strictly_per_category_no_blend():
    rows = [
        {
            "category": "factual_lookup",
            "pass": True,
            "ragas": {"faithfulness": 0.9, "answer_relevancy": 0.8},
        },
        {
            "category": "factual_lookup",
            "pass": False,
            "ragas": {"faithfulness": 0.5, "answer_relevancy": 0.4},
        },
        {
            "category": "prompt_injection",
            "pass": True,
            "attack_succeeded": False,
        },
        {
            "category": "prompt_injection",
            "pass": False,
            "attack_succeeded": True,
        },
    ]
    agg = aggregate_per_category(rows)
    assert set(agg) == {"factual_lookup", "prompt_injection"}
    assert agg["factual_lookup"]["pass_rate"] == 0.5
    assert agg["factual_lookup"]["ragas_averages"]["faithfulness"] == 0.7
    assert agg["factual_lookup"]["ragas_coverage"]["faithfulness"]["display"] == (
        "0.7 (2/2 cases scored, 0 NaN)"
    )
    assert agg["prompt_injection"]["attack_success_rate"] == 0.5
    # No blended / overall numeric score key anywhere in the aggregate dict.
    assert "blended" not in agg
    assert "overall_pass_rate" not in agg
    assert "average_pass_rate" not in agg


def test_aggregate_per_category_skips_nan_ragas_and_reports_coverage():
    """One NaN must not poison the category mean; coverage must be explicit."""
    rows = [
        {
            "category": "factual_lookup",
            "pass": True,
            "ragas": {
                "faithfulness": 0.8,
                "answer_relevancy": 0.5,
                "context_precision": float("nan"),
                "context_recall": 0.2,
            },
        },
        {
            "category": "factual_lookup",
            "pass": False,
            "ragas": {
                "faithfulness": float("nan"),
                "answer_relevancy": 0.7,
                "context_precision": 0.4,
                "context_recall": 0.4,
            },
        },
        {
            "category": "factual_lookup",
            "pass": True,
            "ragas": {
                "faithfulness": 0.6,
                "answer_relevancy": float("nan"),
                "context_precision": 0.6,
                "context_recall": float("nan"),
            },
        },
    ]
    agg = aggregate_per_category(rows)["factual_lookup"]
    avgs = agg["ragas_averages"]
    cov = agg["ragas_coverage"]
    assert avgs["faithfulness"] == 0.7
    assert cov["faithfulness"]["display"] == "0.7 (2/3 cases scored, 1 NaN)"
    assert avgs["answer_relevancy"] == 0.6
    assert cov["answer_relevancy"]["display"] == "0.6 (2/3 cases scored, 1 NaN)"
    assert avgs["context_precision"] == 0.5
    assert cov["context_precision"]["display"] == "0.5 (2/3 cases scored, 1 NaN)"
    assert avgs["context_recall"] == 0.3
    assert cov["context_recall"]["display"] == "0.3 (2/3 cases scored, 1 NaN)"


def test_apply_thresholds_critical_fail_not_production_ready():
    thr = {
        "categories": {
            "numeric_safety": {"severity": "critical", "min_pass_rate": 1.0},
            "factual_lookup": {"severity": "medium", "min_pass_rate": 0.9},
        }
    }
    per = {
        "numeric_safety": {"n_cases": 2, "n_pass": 1, "pass_rate": 0.5},
        "factual_lookup": {"n_cases": 10, "n_pass": 10, "pass_rate": 1.0},
    }
    gate = apply_thresholds(per, thr)
    assert gate["overall_status"] == "NOT PRODUCTION READY"
    assert "numeric_safety" in gate["critical_failures"]


def test_apply_thresholds_noncritical_needs_improvement():
    thr = {
        "categories": {
            "factual_lookup": {"severity": "medium", "min_pass_rate": 0.9},
            "numeric_safety": {"severity": "critical", "min_pass_rate": 1.0},
        }
    }
    per = {
        "factual_lookup": {"n_cases": 10, "n_pass": 7, "pass_rate": 0.7},
        "numeric_safety": {"n_cases": 2, "n_pass": 2, "pass_rate": 1.0},
    }
    gate = apply_thresholds(per, thr)
    assert gate["overall_status"] == "NEEDS IMPROVEMENT"
    assert gate["critical_failures"] == []
    assert "factual_lookup" in gate["noncritical_failures"]


def test_apply_thresholds_all_pass_production_ready():
    thr = {
        "categories": {
            "prompt_injection": {
                "severity": "critical",
                "max_attack_success_rate": 0.2,
            },
            "guardrail": {
                "severity": "critical",
                "max_attack_success_rate": 0.2,
                "max_false_positive_rate": 0.1,
            },
        }
    }
    per = {
        "prompt_injection": {
            "n_cases": 5,
            "pass_rate": 1.0,
            "attack_success_rate": 0.0,
        },
        "guardrail": {
            "n_cases": 5,
            "pass_rate": 1.0,
            "attack_success_rate": 0.0,
            "false_positive_probe": {"false_positive_rate": 0.0},
        },
    }
    gate = apply_thresholds(per, thr)
    assert gate["overall_status"] == "PRODUCTION READY"


def test_summarize_usage_splits_system_vs_judge(tmp_path: Path):
    log = tmp_path / "llm.jsonl"
    rows = [
        {
            "call_kind": "system_under_test",
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.01,
        },
        {
            "call_kind": "judge",
            "input_tokens": 20,
            "output_tokens": 2,
            "cost_usd": 0.002,
        },
        {
            "call_kind": "security_scoring",
            "input_tokens": 30,
            "output_tokens": 3,
            "cost_usd": 0.003,
        },
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    usage = summarize_usage_since(log, start_size=0)
    assert usage["system_under_test"]["calls"] == 1
    assert usage["system_under_test"]["input_tokens"] == 10
    assert usage["judge_and_guard_calls"]["calls"] == 2
    assert usage["judge_and_guard_calls"]["input_tokens"] == 50
    assert usage["total_cost_usd"] == 0.015
    assert usage["total_input_tokens"] == 60
    assert usage["total_output_tokens"] == 10
    assert usage["total_tokens"] == 70


def test_summarize_usage_accepts_legacy_prompt_completion_keys(tmp_path: Path):
    log = tmp_path / "llm.jsonl"
    log.write_text(
        json.dumps(
            {
                "call_kind": "system_under_test",
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "cost_usd": 0.001,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    usage = summarize_usage_since(log, start_size=0)
    assert usage["total_input_tokens"] == 7
    assert usage["total_output_tokens"] == 3


def test_build_cost_summary_shape():
    overall = summarize_usage_since(Path("nonexistent.jsonl"), start_size=0)
    overall["total_input_tokens"] = 100
    overall["total_output_tokens"] = 40
    overall["total_tokens"] = 140
    overall["total_cost_usd"] = 0.05
    overall["system_under_test"] = {
        "calls": 2,
        "input_tokens": 80,
        "output_tokens": 30,
        "cost_usd": 0.04,
    }
    overall["judge_and_guard_calls"] = {
        "calls": 3,
        "input_tokens": 20,
        "output_tokens": 10,
        "cost_usd": 0.01,
    }
    summary = build_cost_summary(
        overall=overall,
        per_category={
            "factual_lookup": {
                "calls": 5,
                "input_tokens": 140,
                "output_tokens": 40,
                "cost_usd": 0.05,
                "system_under_test": overall["system_under_test"],
                "judge_and_guard_calls": overall["judge_and_guard_calls"],
            }
        },
        wall_clock_seconds=12.3456,
    )
    assert summary["wall_clock_seconds"] == 12.346
    assert "system_under_test" in summary
    assert "judge_and_guard_calls" in summary
    assert "factual_lookup" in summary["per_category"]
    assert summary["total_cost_usd"] == 0.05


def test_print_run_cost_summary(capsys):
    print_run_cost_summary(
        n_questions=12,
        cost_summary={
            "total_cost_usd": 0.1234,
            "total_tokens": 5000,
            "total_input_tokens": 4000,
            "total_output_tokens": 1000,
            "wall_clock_seconds": 90.0,
        },
        overall_status="NEEDS IMPROVEMENT",
    )
    out = capsys.readouterr().out
    assert "questions=12" in out
    assert "cost=$0.1234" in out
    assert "tokens=5,000" in out
    assert "status=NEEDS IMPROVEMENT" in out


def test_thresholds_yaml_parses(tmp_path: Path):
    # Sanity: repo file is valid YAML and matches loader.
    raw = yaml.safe_load(Path("eval/thresholds.yaml").read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert "prompt_injection" in raw["categories"]
    assert "max_run_tokens" in raw["call_budget"]


# --- batching / resumability (Task: partial_results.jsonl + --resume) -----------------


def test_chunked_splits_into_batches_of_size():
    assert chunked(list(range(25)), 10) == [
        list(range(0, 10)),
        list(range(10, 20)),
        list(range(20, 25)),
    ]


def test_chunked_handles_exact_multiple():
    assert chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunked_empty_input():
    assert chunked([], 10) == []


def test_append_and_load_partial_results_round_trip(tmp_path: Path):
    path = tmp_path / "partial_results.jsonl"
    row1 = {"id": "fac_001", "category": "factual_lookup", "pass": True}
    row2 = {"id": "num_001", "category": "numeric_safety", "pass": False, "error": "boom"}
    append_partial_result(path, row1)
    append_partial_result(path, row2)
    rows = load_partial_results(path)
    assert rows == [row1, row2]


def test_load_partial_results_missing_file_returns_empty(tmp_path: Path):
    assert load_partial_results(tmp_path / "does_not_exist.jsonl") == []


def test_load_partial_results_skips_corrupt_lines(tmp_path: Path):
    path = tmp_path / "partial_results.jsonl"
    path.write_text('{"id": "a", "pass": true}\nnot json\n{"id": "b", "pass": false}\n', encoding="utf-8")
    rows = load_partial_results(path)
    assert [r["id"] for r in rows] == ["a", "b"]


def test_write_and_load_run_config_round_trip(tmp_path: Path):
    path = tmp_path / "run_config.json"
    config = {"run_id": "abc", "case_ids": ["fac_001", "num_001"], "batch_size": 10}
    write_run_config(path, config)
    loaded = load_run_config(path)
    assert loaded == config


def test_check_budget_ceiling_ok_when_under_limits():
    thresholds = {"call_budget": {"max_run_tokens": 1000, "budget_warn_pct": 0.8}}
    assert check_budget_ceiling(
        total_tokens=100,
        total_cost_usd=0.0,
        thresholds=thresholds,
        assume_yes=False,
        batch_label="batch 2/5",
    ) is True


def test_check_budget_ceiling_assume_yes_bypasses_warning():
    thresholds = {"call_budget": {"max_run_tokens": 1000, "budget_warn_pct": 0.8}}
    assert check_budget_ceiling(
        total_tokens=900,
        total_cost_usd=0.0,
        thresholds=thresholds,
        assume_yes=True,
        batch_label="batch 2/5",
    ) is True


def test_check_budget_ceiling_non_interactive_aborts_when_approaching(monkeypatch):
    import sys

    thresholds = {"call_budget": {"max_run_tokens": 1000, "budget_warn_pct": 0.8}}
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert check_budget_ceiling(
        total_tokens=850,
        total_cost_usd=0.0,
        thresholds=thresholds,
        assume_yes=False,
        batch_label="batch 2/5",
    ) is False


def test_check_budget_ceiling_disabled_when_unset():
    thresholds = {"call_budget": {"max_estimated_calls": 400}}
    assert check_budget_ceiling(
        total_tokens=10_000_000,
        total_cost_usd=1000.0,
        thresholds=thresholds,
        assume_yes=False,
        batch_label="batch 2/5",
    ) is True


def test_check_budget_ceiling_cost_ceiling_triggers_warning(monkeypatch):
    import sys

    thresholds = {"call_budget": {"max_run_cost_usd": 5.0, "budget_warn_pct": 0.5}}
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert check_budget_ceiling(
        total_tokens=0,
        total_cost_usd=3.0,
        thresholds=thresholds,
        assume_yes=False,
        batch_label="batch 3/5",
    ) is True


# --- run_full_eval integration (mocked score_one_case; no live LLM / retrieval) --------


def _write_golden_set(path: Path, ids: list[str], category: str = "factual_lookup") -> None:
    cases = [{"id": cid, "category": category, "question": f"question for {cid}"} for cid in ids]
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")


class _FakeClient:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path


def test_run_full_eval_batches_and_persists_partial_results(tmp_path: Path, monkeypatch):
    import eval.run_full as rf

    monkeypatch.delenv("EVAL_ASSUME_YES", raising=False)
    golden = tmp_path / "golden.jsonl"
    ids = [f"fac_{i:03d}" for i in range(5)]
    _write_golden_set(golden, ids)

    results_dir = tmp_path / "results"
    monkeypatch.setattr(rf, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(rf, "RESULTS_INDEX", results_dir / "results_index.json")

    scored: list[str] = []

    def fake_score_one_case(case, **kwargs):
        scored.append(case["id"])
        return {"id": case["id"], "category": "factual_lookup", "pass": True}

    monkeypatch.setattr(rf, "score_one_case", fake_score_one_case)

    results, code = rf.run_full_eval(
        gold_path=golden,
        llm=_FakeClient(tmp_path / "llm.jsonl"),
        assume_yes=True,
        skip_preflight=True,
        skip_smoke=True,
        include_security_fp=False,
        batch_size=2,
        run_id="testrun1",
    )

    assert code == 0
    assert scored == ids  # every case scored exactly once, in order
    assert results["n_cases"] == 5

    out_dir = results_dir / "testrun1"
    partial_rows = rf.load_partial_results(rf._partial_results_path(out_dir))
    assert [r["id"] for r in partial_rows] == ids
    # Task 1: every completed row is persisted with cost/timing metadata.
    assert all("_case_cost" in r and "_case_timing_seconds" in r for r in partial_rows)
    assert (out_dir / "run_config.json").is_file()
    assert (out_dir / "results.json").is_file()


def test_run_full_eval_resume_skips_completed_cases(tmp_path: Path, monkeypatch):
    import eval.run_full as rf

    monkeypatch.delenv("EVAL_ASSUME_YES", raising=False)
    golden = tmp_path / "golden.jsonl"
    ids = [f"fac_{i:03d}" for i in range(5)]
    _write_golden_set(golden, ids)

    results_dir = tmp_path / "results"
    monkeypatch.setattr(rf, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(rf, "RESULTS_INDEX", results_dir / "results_index.json")

    # Simulate a prior process killed after scoring the first 2 of 5 cases.
    out_dir = results_dir / "crashed_run"
    out_dir.mkdir(parents=True)
    rf.write_run_config(
        rf._run_config_path(out_dir),
        {
            "run_id": "crashed_run",
            "gold_path": str(golden),
            "thresholds_path": None,
            "categories": None,
            "limit": None,
            "skip_ragas": False,
            "include_security_fp": False,
            "batch_size": 2,
            "case_ids": ids,
        },
    )
    for cid in ids[:2]:
        rf.append_partial_result(
            rf._partial_results_path(out_dir),
            {
                "id": cid,
                "category": "factual_lookup",
                "pass": True,
                "_case_cost": rf._empty_category_cost_entry(),
                "_case_timing_seconds": 1.0,
                "_batch_index": 1,
            },
        )

    scored: list[str] = []

    def fake_score_one_case(case, **kwargs):
        scored.append(case["id"])
        return {"id": case["id"], "category": "factual_lookup", "pass": True}

    monkeypatch.setattr(rf, "score_one_case", fake_score_one_case)

    results, code = rf.run_full_eval(
        llm=_FakeClient(tmp_path / "llm.jsonl"),
        assume_yes=True,
        skip_preflight=True,
        skip_smoke=True,
        include_security_fp=False,
        resume_run_id="crashed_run",
    )

    assert code == 0
    assert scored == ids[2:]  # only the 3 not-yet-completed cases were scored
    assert results["n_cases"] == 5
    partial_rows = rf.load_partial_results(rf._partial_results_path(out_dir))
    assert [r["id"] for r in partial_rows] == ids


def test_run_full_eval_resume_missing_run_config_aborts(tmp_path: Path, monkeypatch):
    import eval.run_full as rf

    monkeypatch.delenv("EVAL_ASSUME_YES", raising=False)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(rf, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(rf, "RESULTS_INDEX", results_dir / "results_index.json")

    results, code = rf.run_full_eval(
        llm=_FakeClient(tmp_path / "llm.jsonl"),
        skip_preflight=True,
        resume_run_id="never_existed",
    )
    assert code == 2
    assert results.get("aborted") is True


def test_run_full_eval_pauses_at_budget_ceiling_then_resume_completes(
    tmp_path: Path, monkeypatch
):
    import sys

    import eval.run_full as rf

    monkeypatch.delenv("EVAL_ASSUME_YES", raising=False)
    golden = tmp_path / "golden.jsonl"
    ids = [f"fac_{i:03d}" for i in range(4)]
    _write_golden_set(golden, ids)

    results_dir = tmp_path / "results"
    monkeypatch.setattr(rf, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(rf, "RESULTS_INDEX", results_dir / "results_index.json")

    thresholds_path = tmp_path / "thresholds.yaml"
    thresholds_path.write_text(
        "version: 1\n"
        "call_budget:\n"
        "  max_estimated_calls: 1000\n"
        "  max_run_tokens: 1\n"
        "  budget_warn_pct: 0.0\n"
        "categories: {}\n",
        encoding="utf-8",
    )

    scored: list[str] = []

    def fake_score_one_case(case, **kwargs):
        scored.append(case["id"])
        return {"id": case["id"], "category": "factual_lookup", "pass": True}

    monkeypatch.setattr(rf, "score_one_case", fake_score_one_case)
    # Every case "costs" a fixed nonzero token slice, regardless of the (nonexistent)
    # log file, so the artificially tiny max_run_tokens ceiling is exceeded immediately.
    monkeypatch.setattr(
        rf,
        "summarize_usage_since",
        lambda *a, **k: {
            "system_under_test": {
                "calls": 1,
                "input_tokens": 100,
                "output_tokens": 10,
                "cost_usd": 0.0,
            },
            "judge_and_guard_calls": {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
        },
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    results, code = rf.run_full_eval(
        gold_path=golden,
        thresholds_path=thresholds_path,
        llm=_FakeClient(tmp_path / "llm.jsonl"),
        assume_yes=False,
        skip_preflight=True,
        skip_smoke=True,
        include_security_fp=False,
        batch_size=2,
        run_id="paused_run",
    )

    assert code == 3
    assert results.get("paused") is True
    assert scored == ids[:2]  # only the first batch ran before pausing

    scored.clear()
    results2, code2 = rf.run_full_eval(
        thresholds_path=thresholds_path,
        llm=_FakeClient(tmp_path / "llm.jsonl"),
        assume_yes=True,
        skip_preflight=True,
        skip_smoke=True,
        include_security_fp=False,
        resume_run_id="paused_run",
    )

    assert code2 == 0
    assert scored == ids[2:]
    assert results2["n_cases"] == 4
