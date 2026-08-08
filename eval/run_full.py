"""Full eval orchestration: Steps 4–6 scorers → per-category gates → results.

Canonical golden set: ``eval/golden_set.jsonl`` (**30** cases). The prior 152-case
set is archived at ``eval/golden_set_152_archive.jsonl``. Pre-30 run artifacts live
under ``eval/results_archive_pre30/`` (not ``eval/results/``).

Usage::

    python -m eval.run_full                  # preflight -> smoke (5 cases) -> confirm -> full run
    python -m eval.run_full --smoke-only      # just the 5-case smoke subset, then exit
    python -m eval.run_full --skip-smoke      # skip the smoke subset (not recommended)
    python -m eval.run_full --batch-size 5    # smaller batches (default 10)
    python -m eval.run_full --resume 20260805T160000Z   # continue an interrupted run
    python -m eval.run_full --yes --limit 2
    python -m eval.run_full --skip-ragas --skip-security-fp
    python -m eval.run_full --resume-scoring-only RUN_ID
    python -m eval.run_full --only-failed RUN_ID
    python -m eval.run_full --fill-gaps RUN_ID  # missing + failed vs golden_set
    python -m eval.run_full --gap-set RUN_ID    # alias for --fill-gaps
    python -m eval.run_full --category cross_regulation,design_implication

The golden set is processed in batches (default 10 cases; ``--batch-size``).
After EVERY case — not just every batch — the result row is appended
immediately to ``eval/results/{run_id}/partial_results.jsonl`` (flushed +
fsync'd), so a killed or hung process never loses already-scored work.
``--resume {run_id}`` reads that file, skips already-completed case ids, and
continues from the next batch. Before starting each batch (after the first),
accumulated cost/tokens are checked against ``thresholds.yaml``'s
``call_budget`` ceiling; if approached, the run pauses for confirmation
instead of silently continuing into a provider's daily quota.

Derivative modes (``--resume-scoring-only``, ``--only-failed``, ``--fill-gaps`` /
``--gap-set``) always write a NEW ``run_id`` directory that references
``parent_run_id`` — the original run is never overwritten. Cases not re-processed
keep their prior rows; aggregation is recomputed on the merged set. ``--fill-gaps``
covers every golden case that is missing from the parent run **or** has
``pass=False`` (including unrecoverable error rows). Interrupted fill-gaps runs
resume with ``--resume {derivative_run_id}`` (same ``partial_results.jsonl``
checkpointing as a full run).

Aggregation is STRICTLY per category. There is intentionally no blended /
overall numeric score across categories — averaging risk-heterogeneous
categories (e.g. factual_lookup with prompt_injection) hides critical failures.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

# DeepTeam (Python 3.13) still imports removed stdlib nntplib — shim before scorers.
from eval.nntplib_shim import ensure_nntplib_shim

ensure_nntplib_shim()

from eval.gold import DEFAULT_GOLDEN, load_golden_set
from eval.preflight_check import confirm_or_abort as _confirm_preflight
from eval.preflight_check import format_table as _format_preflight_table
from eval.preflight_check import run_preflight
from eval.aggregation import (
    DEFAULT_THRESHOLDS,
    GapMergeError,
    aggregate_per_category,
    apply_thresholds,
    compute_gap_set,
    load_partial_results,
    load_thresholds,
    merge_fill_gaps_results,
)
from eval.case_scoring import (
    NUMERIC_ONLY,
    OUT_OF_SCOPE,
    _add_bucket,
    _category_cost_entry,
    _empty_token_bucket,
    _usage_snapshot,
    format_scoring_provider_summary,
    rescore_saved_case,
    score_one_case,
    summarize_scoring_providers_since,
    summarize_usage_since,
)
from eval.case_timeout import (
    case_timeout_enabled,
    score_case_with_timeout,
    shutdown_case_timeout_pool,
)
from eval.scoring.ragas_scorer import RAGAS_SCORE_CATEGORIES
from eval.scoring.security_scorer import (
    SECURITY_CATEGORIES,
    measure_guardrail_false_positive_rate,
)
from eval.smoke_subset import confirm_smoke_or_abort, print_smoke_summary, run_smoke_subset
from generation.llm_client import LLMClient

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_INDEX = RESULTS_DIR / "results_index.json"

# Cases per batch; results are persisted to partial_results.jsonl after each one.
DEFAULT_BATCH_SIZE = 10


def _score_case_row(
    case: dict[str, Any],
    *,
    llm: LLMClient,
    skip_ragas: bool,
    security_judge: Any,
    guard_input: Any,
    guard_output: Any,
) -> dict[str, Any]:
    """Score one case; process-isolated wall-clock deadline when enabled.

    ``EVAL_CASE_TIMEOUT=0`` keeps the historical in-process path (shared LLM /
    judge objects). Default is a persistent subprocess worker so a hung native
    call can be killed without AV'ing the parent harness (see ``eval.case_timeout``).
    """
    if case_timeout_enabled():
        return score_case_with_timeout(
            case,
            skip_ragas=skip_ragas,
            with_security=security_judge is not None,
            llm_provider=str(getattr(llm, "provider", None) or "") or None,
        )
    return score_one_case(
        case,
        llm=llm,
        skip_ragas=skip_ragas,
        security_judge=security_judge,
        guard_input=guard_input,
        guard_output=guard_output,
    )

# Rough per-case LLM call lower bounds (system / judge / guard). Rewrite optional.
_ESTIMATE_PER_CASE: dict[str, dict[str, int]] = {
    # RAGAS: answer (+rewrite) + ~4 judge metric calls
    "factual_lookup": {"system": 1, "judge": 4, "guard": 0},
    "compliance_check": {"system": 1, "judge": 4, "guard": 0},
    "multi_hop": {"system": 1, "judge": 4, "guard": 0},
    "enumerative": {"system": 1, "judge": 4, "guard": 0},
    "cross_regulation": {"system": 1, "judge": 4, "guard": 0},
    "design_implication": {"system": 1, "judge": 4, "guard": 0},
    "numeric_safety": {"system": 1, "judge": 0, "guard": 0},
    "out_of_scope": {"system": 1, "judge": 0, "guard": 0},
    # DeepEval HallucinationMetric ≈ 1–2 judge calls
    "hallucination_probe": {"system": 1, "judge": 2, "guard": 0},
    # Toxicity + Privacy + Topical on input and output
    "guardrail": {"system": 1, "judge": 0, "guard": 6},
    # PromptInjection.enhance (~2–4) + 2 PromptInjectionGuard calls
    "prompt_injection": {"system": 1, "judge": 4, "guard": 2},
}

# Legitimate FP probes run once when scoring guardrail (see security_scorer).
_FP_PROBE_COUNT = 8
_FP_CALLS_PER_PROBE = {"system": 1, "judge": 0, "guard": 6}


def estimate_run_llm_calls(
    cases: Sequence[dict[str, Any]],
    *,
    rewrite_enabled: bool = True,
    skip_ragas: bool = False,
    include_security_fp: bool = True,
    provider: str | None = None,
) -> dict[str, Any]:
    """Estimate system + judge + guard LLM calls before a full run."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "mock").strip().lower()
    by_cat: dict[str, int] = defaultdict(int)
    for c in cases:
        cat = str(c.get("category") or "").strip().lower()
        by_cat[cat] += 1

    system = judge = guard = 0
    breakdown: dict[str, dict[str, int]] = {}
    for cat, n in sorted(by_cat.items()):
        base = dict(_ESTIMATE_PER_CASE.get(cat, {"system": 1, "judge": 0, "guard": 0}))
        if cat in RAGAS_SCORE_CATEGORIES and skip_ragas:
            base["judge"] = 0
        if rewrite_enabled and cat in (
            RAGAS_SCORE_CATEGORIES | NUMERIC_ONLY | OUT_OF_SCOPE
        ):
            # RAGAS / custom-only live path uses rewrite=True.
            base["system"] = base.get("system", 1) + 1
        if (os.getenv("SECURITY_SKIP_ATTACK_ENHANCE") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        } and cat == "prompt_injection":
            base["judge"] = max(0, base.get("judge", 0) - 3)

        s, j, g = base.get("system", 0) * n, base.get("judge", 0) * n, base.get("guard", 0) * n
        system += s
        judge += j
        guard += g
        breakdown[cat] = {"n_cases": n, "system": s, "judge": j, "guard": g, "subtotal": s + j + g}

    fp_block: dict[str, Any] | None = None
    if include_security_fp and by_cat.get("guardrail", 0) > 0:
        fp_s = _FP_CALLS_PER_PROBE["system"] * _FP_PROBE_COUNT
        fp_g = _FP_CALLS_PER_PROBE["guard"] * _FP_PROBE_COUNT
        system += fp_s
        guard += fp_g
        fp_block = {
            "n_probes": _FP_PROBE_COUNT,
            "system": fp_s,
            "judge": 0,
            "guard": fp_g,
            "subtotal": fp_s + fp_g,
            "note": "Guardrail false-positive probe (legitimate questions)",
        }

    total = system + judge + guard
    note = (
        "Lower-bound estimate; RAGAS/DeepEval/DeepTeam may issue extra judge/guard calls. "
        f"LLM_PROVIDER={provider}."
    )
    if provider == "mock":
        note += " Mock provider — no billable remote quota, but call count still estimated."

    return {
        "system_calls": system,
        "judge_calls": judge,
        "guard_calls": guard,
        "total": total,
        "by_category": breakdown,
        "false_positive_probe": fp_block,
        "provider": provider,
        "note": note,
    }


def _confirm_call_budget(
    estimate: dict[str, Any],
    *,
    max_estimated_calls: int,
    assume_yes: bool,
) -> bool:
    total = int(estimate.get("total") or 0)
    print(
        f"Estimated LLM calls: {total} "
        f"(system={estimate['system_calls']}, "
        f"judge={estimate['judge_calls']}, "
        f"guard={estimate['guard_calls']}) "
        f"[safety threshold={max_estimated_calls}]"
    )
    print(estimate.get("note") or "")
    if total <= max_estimated_calls:
        return True
    if assume_yes:
        print(
            f"Estimate {total} exceeds safety threshold {max_estimated_calls}; "
            "proceeding because --yes / EVAL_ASSUME_YES was set.",
            file=sys.stderr,
        )
        return True
    if not sys.stdin.isatty():
        print(
            f"Estimate {total} exceeds safety threshold {max_estimated_calls} "
            "and stdin is not a TTY. Pass --yes to proceed.",
            file=sys.stderr,
        )
        return False
    reply = input(
        f"About to make ~{total} LLM calls (>{max_estimated_calls}). Continue? [y/N] "
    ).strip().lower()
    return reply in {"y", "yes"}


def _new_cat_usage_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "system_under_test": _empty_token_bucket(),
        "judge_and_guard_calls": _empty_token_bucket(),
    }

def _empty_category_cost_entry() -> dict[str, Any]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "system_under_test": _empty_token_bucket(),
        "judge_and_guard_calls": _empty_token_bucket(),
    }

def _accumulate_case_cost(bucket: dict[str, Any], entry: dict[str, Any]) -> None:
    """Fold one case's ``_category_cost_entry``-shaped cost into a running bucket."""
    bucket["calls"] += int(entry.get("calls") or 0)
    bucket["input_tokens"] += int(entry.get("input_tokens") or 0)
    bucket["output_tokens"] += int(entry.get("output_tokens") or 0)
    bucket["cost_usd"] = round(float(bucket.get("cost_usd") or 0.0) + float(entry.get("cost_usd") or 0.0), 8)
    _add_bucket(bucket["system_under_test"], entry.get("system_under_test") or _empty_token_bucket())
    _add_bucket(
        bucket["judge_and_guard_calls"], entry.get("judge_and_guard_calls") or _empty_token_bucket()
    )

def _overall_usage_from_per_category(per_cat_usage: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Running-total usage summed from the per-category buckets we maintain in memory.

    Deliberately NOT derived from a Portkey-log file-offset slice: that offset is only
    meaningful within a single process, and this total must stay correct across
    ``--resume`` (i.e. across process restarts), where it's seeded from
    ``partial_results.jsonl`` instead.
    """
    sut = _empty_token_bucket()
    jg = _empty_token_bucket()
    for bucket in per_cat_usage.values():
        _add_bucket(sut, bucket["system_under_test"])
        _add_bucket(jg, bucket["judge_and_guard_calls"])
    total_in = sut["input_tokens"] + jg["input_tokens"]
    total_out = sut["output_tokens"] + jg["output_tokens"]
    return {
        "system_under_test": sut,
        "judge_and_guard_calls": jg,
        "judge_and_guard": jg,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "total_cost_usd": round(sut["cost_usd"] + jg["cost_usd"], 8),
    }


def build_cost_summary(
    *,
    overall: dict[str, Any],
    per_category: dict[str, dict[str, Any]],
    wall_clock_seconds: float,
) -> dict[str, Any]:
    """Assemble the ``cost_summary`` block for results.json."""
    return {
        "total_input_tokens": int(overall.get("total_input_tokens") or 0),
        "total_output_tokens": int(overall.get("total_output_tokens") or 0),
        "total_tokens": int(overall.get("total_tokens") or 0),
        "total_cost_usd": float(overall.get("total_cost_usd") or 0.0),
        "wall_clock_seconds": round(float(wall_clock_seconds), 3),
        "system_under_test": dict(overall.get("system_under_test") or _empty_token_bucket()),
        "judge_and_guard_calls": dict(
            overall.get("judge_and_guard_calls") or _empty_token_bucket()
        ),
        "per_category": per_category,
    }


def print_run_cost_summary(
    *,
    n_questions: int,
    cost_summary: dict[str, Any],
    overall_status: str,
) -> None:
    """Short console summary at end of run."""
    dur = float(cost_summary.get("wall_clock_seconds") or 0.0)
    if dur >= 3600:
        dur_s = f"{dur / 3600:.2f}h"
    elif dur >= 60:
        dur_s = f"{dur / 60:.1f}m"
    else:
        dur_s = f"{dur:.1f}s"
    cost = float(cost_summary.get("total_cost_usd") or 0.0)
    tokens = int(cost_summary.get("total_tokens") or 0)
    print(
        f"SUMMARY  questions={n_questions}  "
        f"cost=${cost:.4f}  "
        f"tokens={tokens:,}  "
        f"(in={int(cost_summary.get('total_input_tokens') or 0):,} / "
        f"out={int(cost_summary.get('total_output_tokens') or 0):,})  "
        f"duration={dur_s}  "
        f"status={overall_status}"
    )


def _append_results_index(entry: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_INDEX.is_file():
        try:
            idx = json.loads(RESULTS_INDEX.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            idx = {"version": 1, "runs": []}
    else:
        idx = {
            "version": 1,
            "description": "Summary index of timestamped eval runs under eval/results/.",
            "runs": [],
        }
    runs = list(idx.get("runs") or [])
    runs.append(entry)
    idx["runs"] = runs
    idx["version"] = int(idx.get("version") or 1)
    RESULTS_INDEX.write_text(
        json.dumps(idx, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def chunked(seq: Sequence[Any], size: int) -> list[list[Any]]:
    """Split ``seq`` into consecutive chunks of at most ``size`` items."""
    size = max(1, int(size))
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def _partial_results_path(out_dir: Path) -> Path:
    return out_dir / "partial_results.jsonl"


def _run_config_path(out_dir: Path) -> Path:
    return out_dir / "run_config.json"


def append_partial_result(path: Path, row: dict[str, Any]) -> None:
    """Append one completed case immediately — the resumability mechanism.

    Flushed + fsync'd so a killed/hung process cannot lose already-scored cases;
    everything scored so far stays safely on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_run_config(path: Path, config: dict[str, Any]) -> None:
    """Freeze the exact case list/options for this run so ``--resume`` replays it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_run_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _budget_ceiling_config(thresholds: dict[str, Any]) -> dict[str, Any]:
    cb = thresholds.get("call_budget") or {}

    def _env_or(key: str, env: str) -> Any:
        raw = os.getenv(env)
        if raw is not None and raw.strip():
            return raw.strip()
        return cb.get(key)

    max_tokens = _env_or("max_run_tokens", "EVAL_MAX_RUN_TOKENS")
    max_cost = _env_or("max_run_cost_usd", "EVAL_MAX_RUN_COST_USD")
    warn_pct = _env_or("budget_warn_pct", "EVAL_BUDGET_WARN_PCT")
    return {
        "max_tokens": int(max_tokens) if max_tokens not in (None, "") else None,
        "max_cost_usd": float(max_cost) if max_cost not in (None, "") else None,
        "warn_pct": float(warn_pct) if warn_pct not in (None, "") else 0.8,
    }


def check_budget_ceiling(
    *,
    total_tokens: int,
    total_cost_usd: float,
    thresholds: dict[str, Any],
    assume_yes: bool,
    batch_label: str,
) -> bool:
    """Pause for confirmation once cumulative tokens/cost approach the configured ceiling.

    Checked BEFORE each batch after the first, so a Groq/NIM free-tier daily quota
    doesn't get exceeded silently mid-batch (a 429 storm) — the run pauses at a clean
    batch boundary with a ``--resume`` point instead.
    """
    cfg = _budget_ceiling_config(thresholds)
    warnings: list[str] = []
    if cfg["max_tokens"]:
        pct = total_tokens / cfg["max_tokens"]
        if pct >= cfg["warn_pct"]:
            warnings.append(f"tokens {total_tokens:,}/{cfg['max_tokens']:,} ({pct:.0%} of ceiling)")
    if cfg["max_cost_usd"]:
        pct = total_cost_usd / cfg["max_cost_usd"]
        if pct >= cfg["warn_pct"]:
            warnings.append(
                f"cost ${total_cost_usd:.4f}/${cfg['max_cost_usd']:.2f} ({pct:.0%} of ceiling)"
            )
    if not warnings:
        return True

    print(
        f"\nWARNING: approaching configured budget ceiling before {batch_label}: "
        + "; ".join(warnings),
        file=sys.stderr,
    )
    if assume_yes:
        print("Proceeding anyway (--yes / EVAL_ASSUME_YES set).", file=sys.stderr)
        return True
    if not sys.stdin.isatty():
        print(
            "stdin is not a TTY — pass --yes to proceed despite the budget warning, "
            "or stop here and continue later with --resume.",
            file=sys.stderr,
        )
        return False
    try:
        reply = input("Continue to next batch anyway? [y/N] ").strip().lower()
    except EOFError:
        reply = ""
    return reply in {"y", "yes"}


def run_full_eval(
    *,
    gold_path: Path | None = None,
    thresholds_path: Path | None = None,
    llm: LLMClient | None = None,
    assume_yes: bool = False,
    skip_ragas: bool = False,
    include_security_fp: bool = True,
    limit: int | None = None,
    categories: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    run_id: str | None = None,
    skip_preflight: bool = False,
    smoke_only: bool = False,
    skip_smoke: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume_run_id: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Execute full suite; return (results_dict, exit_code).

    Default flow (matches ``python -m eval.run_full`` with no flags): preflight
    ping → 5-case smoke subset (full pipeline) → confirm → full batched run.
    ``smoke_only=True`` stops after the smoke subset; ``skip_smoke=True`` skips
    it entirely (not recommended — see ``eval/smoke_subset.py``).

    The golden set is processed in batches of ``batch_size`` cases. Every case
    (not just every batch) is appended immediately to
    ``eval/results/{run_id}/partial_results.jsonl`` as soon as it's scored —
    the resumability mechanism. Pass ``resume_run_id`` to continue an
    interrupted run: already-completed case ids are skipped, and the exact
    case list/options from the original run are replayed from
    ``run_config.json`` (frozen at the start of the original run).
    """
    assume = assume_yes or (os.getenv("EVAL_ASSUME_YES") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    resuming = bool(resume_run_id)
    if resuming and smoke_only:
        print("--smoke-only cannot be combined with --resume.", file=sys.stderr)
        return {"aborted": True, "reason": "smoke_only with resume"}, 2

    if not skip_preflight:
        print("Preflight: pinging each Portkey provider directly (bypassing fallback)...\n")
        preflight_results = run_preflight()
        print(_format_preflight_table(preflight_results))
        if not _confirm_preflight(preflight_results, assume_yes=assume):
            print("Aborted (preflight).", file=sys.stderr)
            return {"aborted": True, "preflight": preflight_results}, 2
        print()

    client = llm or LLMClient()

    # --- Resolve run_id / results dir, and (if resuming) the frozen run config ---
    run_config: dict[str, Any] = {}
    if resuming:
        rid = str(resume_run_id)
        out_dir = RESULTS_DIR / rid
        config_path = _run_config_path(out_dir)
        if not config_path.is_file():
            print(
                f"Aborted: no run_config.json found for --resume {rid!r} under {out_dir} "
                "(nothing to resume).",
                file=sys.stderr,
            )
            return {"aborted": True, "reason": f"missing run_config for {rid}"}, 2
        run_config = load_run_config(config_path)
        gold_path = Path(run_config["gold_path"]) if run_config.get("gold_path") else gold_path
        thresholds_path = (
            Path(run_config["thresholds_path"])
            if run_config.get("thresholds_path")
            else thresholds_path
        )
        categories = run_config.get("categories")
        case_ids = run_config.get("case_ids") or case_ids
        limit = run_config.get("limit")
        skip_ragas = bool(run_config.get("skip_ragas", skip_ragas))
        include_security_fp = bool(run_config.get("include_security_fp", include_security_fp))
        batch_size = int(run_config.get("batch_size") or batch_size)
    else:
        rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = RESULTS_DIR / rid
        # A fresh run (even one reusing an old --run-id) starts with a clean slate.
        stale_partial = _partial_results_path(out_dir)
        if stale_partial.is_file():
            logger.warning("Fresh run reusing run_id=%s: clearing stale partial_results.jsonl", rid)
            stale_partial.unlink()
        stale_config = _run_config_path(out_dir)
        if stale_config.is_file():
            stale_config.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = _run_config_path(out_dir)
    partial_path = _partial_results_path(out_dir)

    if resuming:
        print(f"Resuming run {rid} — skipping smoke subset (already validated on the original run).\n")
    elif not skip_smoke:
        print(
            "Smoke subset: running the FULL pipeline (retrieval, generation, RAGAS, "
            "custom checks, DeepEval/DeepTeam) on 5 representative cases...\n"
        )
        smoke_summary = run_smoke_subset(
            gold_path=gold_path, llm=client, skip_ragas=skip_ragas
        )
        print_smoke_summary(smoke_summary)
        if smoke_only:
            return {"smoke_only": True, "smoke": smoke_summary}, (
                0 if smoke_summary["all_passed"] else 1
            )
        if not confirm_smoke_or_abort(smoke_summary, assume_yes=assume):
            print("Aborted (smoke).", file=sys.stderr)
            return {"aborted": True, "smoke": smoke_summary}, 2
        print()
    elif smoke_only:
        print("--smoke-only requested but --skip-smoke also set; nothing to run.", file=sys.stderr)
        return {"aborted": True, "reason": "smoke_only with skip_smoke"}, 2

    thresholds = load_thresholds(thresholds_path or DEFAULT_THRESHOLDS)

    if resuming:
        all_golden = load_golden_set(gold_path or DEFAULT_GOLDEN)
        by_id = {str(c.get("id")): c for c in all_golden}
        case_id_order = [str(cid) for cid in (run_config.get("case_ids") or [])]
        cases: list[dict[str, Any]] = []
        missing_ids: list[str] = []
        for cid in case_id_order:
            c = by_id.get(cid)
            if c is None:
                missing_ids.append(cid)
            else:
                cases.append(c)
        if missing_ids:
            logger.warning(
                "Resume %s: %d case id(s) from the original run are no longer in the "
                "golden set and will be skipped: %s",
                rid,
                len(missing_ids),
                ", ".join(missing_ids[:10]) + (" ..." if len(missing_ids) > 10 else ""),
            )
    else:
        cases = load_golden_set(gold_path or DEFAULT_GOLDEN)
        if categories:
            wanted = {c.strip().lower() for c in categories if c.strip()}
            cases = [
                c
                for c in cases
                if str(c.get("category") or "").strip().lower() in wanted
            ]
        if case_ids:
            wanted_ids = {str(x).strip() for x in case_ids if str(x).strip()}
            cases = [c for c in cases if str(c.get("id") or "") in wanted_ids]
            missing = wanted_ids - {str(c.get("id") or "") for c in cases}
            if missing:
                logger.warning(
                    "case-id filter: %d id(s) not in golden set: %s",
                    len(missing),
                    ", ".join(sorted(missing)),
                )
        if limit is not None:
            cases = cases[: max(0, int(limit))]

    completed_rows = load_partial_results(partial_path)
    completed_ids: set[str] = {str(r.get("id")) for r in completed_rows}
    remaining_cases = [c for c in cases if str(c.get("id")) not in completed_ids]

    if resuming:
        print(
            f"Resume {rid}: {len(cases)} total cases in this run, {len(completed_rows)} "
            f"already completed, {len(remaining_cases)} remaining.\n"
        )
    else:
        write_run_config(
            config_path,
            {
                "run_id": rid,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "gold_path": str(gold_path or DEFAULT_GOLDEN),
                "thresholds_path": str(thresholds_path or DEFAULT_THRESHOLDS),
                "categories": list(categories) if categories else None,
                "limit": limit,
                "skip_ragas": skip_ragas,
                "include_security_fp": include_security_fp,
                "batch_size": batch_size,
                "case_ids": [c.get("id") for c in cases],
            },
        )

    rewrite_enabled = (os.getenv("RETRIEVAL_REWRITE") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    estimate = estimate_run_llm_calls(
        remaining_cases,
        rewrite_enabled=rewrite_enabled,
        skip_ragas=skip_ragas,
        include_security_fp=include_security_fp
        and any(
            str(c.get("category") or "").lower() == "guardrail" for c in remaining_cases
        ),
    )
    budget = (thresholds.get("call_budget") or {}).get("max_estimated_calls", 400)
    try:
        budget = int(os.getenv("EVAL_MAX_ESTIMATED_CALLS") or budget)
    except ValueError:
        budget = int(budget)

    if not _confirm_call_budget(estimate, max_estimated_calls=budget, assume_yes=assume):
        print("Aborted.", file=sys.stderr)
        return {"aborted": True, "estimate": estimate}, 2

    log_path = Path(client.log_path)
    wall_t0 = time.perf_counter()

    security_judge = None
    guard_in = guard_out = None
    need_security = any(
        str(c.get("category") or "").strip().lower() in SECURITY_CATEGORIES
        for c in remaining_cases
    )
    if need_security:
        from eval.scoring.security_scorer import (
            _build_topic_guards,
            _portkey_security_judge,
        )

        security_judge = _portkey_security_judge(client)
        guard_in, guard_out = _build_topic_guards(security_judge)

    # Per-category cost accumulation — seeded from already-completed rows (resume-safe;
    # see _overall_usage_from_per_category for why this replaces log-offset slicing).
    per_cat_usage: dict[str, dict[str, Any]] = defaultdict(_new_cat_usage_bucket)
    prior_processing_seconds = 0.0
    for row in completed_rows:
        cat = str(row.get("category") or "unknown").strip().lower()
        _accumulate_case_cost(
            per_cat_usage[cat], row.get("_case_cost") or _empty_category_cost_entry()
        )
        prior_processing_seconds += float(row.get("_case_timing_seconds") or 0.0)

    case_results: list[dict[str, Any]] = list(completed_rows)

    global_batches = chunked(cases, batch_size)
    n_total_batches = len(global_batches)
    n_total_cases = len(cases)

    for batch_idx, batch in enumerate(global_batches, 1):
        todo = [c for c in batch if str(c.get("id")) not in completed_ids]
        if not todo:
            continue  # entire batch already completed in a previous session

        batch_log_start = _usage_snapshot(log_path)
        for case in todo:
            cat = str(case.get("category") or "").strip().lower() or "unknown"
            logger.info(
                "[case %s/%s] scoring %s (%s) [batch %s/%s]",
                len(completed_ids) + 1,
                n_total_cases,
                case.get("id"),
                cat,
                batch_idx,
                n_total_batches,
            )
            case_t0 = time.perf_counter()
            case_log_start = _usage_snapshot(log_path)
            try:
                row = _score_case_row(
                    case,
                    llm=client,
                    skip_ragas=skip_ragas,
                    security_judge=security_judge,
                    guard_input=guard_in,
                    guard_output=guard_out,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("case %s failed", case.get("id"))
                row = {
                    "id": case.get("id"),
                    "category": cat,
                    "severity": case.get("severity"),
                    "pass": False,
                    "error": str(exc),
                }
            case_log_end = _usage_snapshot(log_path)
            slice_usage = summarize_usage_since(
                log_path, start_size=case_log_start, end_size=case_log_end
            )
            entry = _category_cost_entry(slice_usage)
            row["_case_cost"] = entry
            row["_case_timing_seconds"] = round(time.perf_counter() - case_t0, 3)
            row["_batch_index"] = batch_idx

            _accumulate_case_cost(per_cat_usage[cat], entry)
            case_results.append(row)
            completed_ids.add(str(case.get("id")))
            # Task 1: persist IMMEDIATELY — one line per completed case, no buffering.
            append_partial_result(partial_path, row)

        batch_log_end = _usage_snapshot(log_path)
        scoring_providers = summarize_scoring_providers_since(
            log_path, start_size=batch_log_start, end_size=batch_log_end
        )
        provider_line = format_scoring_provider_summary(scoring_providers)
        # Persist per-batch provider breakdown for post-run overflow analysis.
        provider_log = out_dir / "scoring_providers_by_batch.jsonl"
        with provider_log.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "batch_index": batch_idx,
                        "n_batches_total": n_total_batches,
                        "cases_in_batch": [str(c.get("id")) for c in todo],
                        "cases_completed_total": sum(
                            1 for c in cases if str(c.get("id")) in completed_ids
                        ),
                        **scoring_providers,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        done_so_far = sum(1 for c in cases if str(c.get("id")) in completed_ids)
        running = _overall_usage_from_per_category(per_cat_usage)
        print(
            f"Batch {batch_idx}/{n_total_batches} complete ({done_so_far}/{n_total_cases} cases). "
            f"Cost so far: ${running['total_cost_usd']:.4f}, {running['total_tokens']:,} tokens. "
            f"{provider_line}. Continuing..."
        )
        if batch_idx < n_total_batches:
            if not check_budget_ceiling(
                total_tokens=running["total_tokens"],
                total_cost_usd=running["total_cost_usd"],
                thresholds=thresholds,
                assume_yes=assume,
                batch_label=f"batch {batch_idx + 1}/{n_total_batches}",
            ):
                print(
                    f"Paused after batch {batch_idx}/{n_total_batches} "
                    f"({done_so_far}/{n_total_cases} cases). Resume with:\n"
                    f"  python -m eval.run_full --resume {rid}",
                    file=sys.stderr,
                )
                return {
                    "paused": True,
                    "run_id": rid,
                    "batches_completed": batch_idx,
                    "cases_completed": done_so_far,
                }, 3

    fp_report: dict[str, Any] | None = None
    if include_security_fp and any(
        str(c.get("category") or "").lower() == "guardrail" for c in cases
    ):
        logger.info("Running guardrail false-positive probe")
        fp_log_start = _usage_snapshot(log_path)
        fp_report = measure_guardrail_false_positive_rate(
            llm=client, judge=security_judge
        )
        fp_log_end = _usage_snapshot(log_path)
        fp_slice = summarize_usage_since(
            log_path, start_size=fp_log_start, end_size=fp_log_end
        )
        fp_cost = _category_cost_entry(fp_slice)
        # Attribute FP probe spend to guardrail (separate sub-key for clarity).
        g = per_cat_usage["guardrail"]
        _accumulate_case_cost(g, fp_cost)
        g["false_positive_probe"] = fp_cost

    wall_clock_seconds = prior_processing_seconds + (time.perf_counter() - wall_t0)

    # Task 5: merge partial_results.jsonl into the final results.json — case_results
    # already IS that merge (prior-session rows + this-session rows), so the same
    # STRICTLY-per-category aggregation just runs over the combined list.
    per_category = aggregate_per_category(
        case_results, false_positive_probe=fp_report
    )
    gate = apply_thresholds(per_category, thresholds)
    overall_usage = _overall_usage_from_per_category(per_cat_usage)
    cost_summary = build_cost_summary(
        overall=overall_usage,
        per_category={k: dict(v) for k, v in sorted(per_cat_usage.items())},
        wall_clock_seconds=wall_clock_seconds,
    )

    ts = datetime.now(timezone.utc)

    results: dict[str, Any] = {
        "run_id": rid,
        "timestamp": ts.isoformat(),
        "golden_set": str(gold_path or DEFAULT_GOLDEN),
        "thresholds": str(thresholds_path or DEFAULT_THRESHOLDS),
        "n_cases": len(case_results),
        "estimate": estimate,
        # Per-case raw results
        "cases": case_results,
        # Per-category aggregates only — no blended cross-category score.
        "per_category": per_category,
        "gate": {
            "overall_status": gate["overall_status"],
            "critical_failures": gate["critical_failures"],
            "noncritical_failures": gate["noncritical_failures"],
            "categories": gate["categories"],
        },
        "overall_status": gate["overall_status"],
        # Primary cost block (Portkey usage logger — no second logging system).
        "cost_summary": cost_summary,
        "false_positive_probe": fp_report,
    }

    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    latest = RESULTS_DIR / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "run_id": rid,
                "path": str(results_path.as_posix()),
                "overall_status": gate["overall_status"],
                "timestamp": ts.isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _append_results_index(
        {
            "run_id": rid,
            "timestamp": ts.isoformat(),
            "path": str(results_path.as_posix()),
            "overall_status": gate["overall_status"],
            "n_cases": len(case_results),
            "total_cost_usd": cost_summary["total_cost_usd"],
            "total_tokens": cost_summary["total_tokens"],
            "wall_clock_seconds": cost_summary["wall_clock_seconds"],
            "critical_failures": gate["critical_failures"],
            "noncritical_failures": gate["noncritical_failures"],
            # Index stores per-category pass_rate / ASR only — never a blend.
            "per_category_summary": {
                cat: {
                    "pass_rate": agg.get("pass_rate"),
                    "attack_success_rate": agg.get("attack_success_rate"),
                    "n_cases": agg.get("n_cases"),
                    "cost_usd": (cost_summary["per_category"].get(cat) or {}).get(
                        "cost_usd"
                    ),
                }
                for cat, agg in per_category.items()
            },
        }
    )

    print(f"Wrote {results_path}")
    for cat, info in gate["categories"].items():
        mark = "OK" if info["met_threshold"] else "FAIL"
        print(f"  [{mark}] {cat} ({info['severity']}): {info.get('failures') or 'met'}")
    print_run_cost_summary(
        n_questions=len(case_results),
        cost_summary=cost_summary,
        overall_status=gate["overall_status"],
    )

    try:
        from eval.render_dashboard import render_dashboard

        dash_path = render_dashboard(results, out_path=out_dir / "dashboard.png")
        print(f"Wrote {dash_path}")
    except Exception as exc:  # noqa: BLE001 — dashboard must not fail the gate
        logger.exception("Dashboard render failed: %s", exc)
        print(f"Dashboard render failed: {exc}", file=sys.stderr)

    exit_code = 1 if gate["overall_status"] == "NOT PRODUCTION READY" else 0
    return results, exit_code


def normalize_category_filters(raw: Sequence[str] | None) -> list[str] | None:
    """Accept repeatable ``--category`` and/or comma-separated names."""
    if not raw:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        for part in str(item).split(","):
            name = part.strip().lower()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out or None


def normalize_case_id_filters(raw: Sequence[str] | None) -> list[str] | None:
    """Accept repeatable ``--case-id`` and/or comma-separated ids."""
    if not raw:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        for part in str(item).split(","):
            cid = part.strip()
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out or None


def load_parent_run_payload(parent_run_id: str) -> dict[str, Any]:
    """Load cases (+ optional FP probe / cost) from results.json or partial_results.jsonl."""
    out_dir = RESULTS_DIR / str(parent_run_id)
    results_path = out_dir / "results.json"
    if results_path.is_file():
        data = json.loads(results_path.read_text(encoding="utf-8"))
        cases = list(data.get("cases") or [])
        if not cases:
            raise ValueError(f"Parent run {parent_run_id} results.json has no cases")
        return {
            "parent_run_id": str(parent_run_id),
            "path": str(results_path),
            "cases": cases,
            "false_positive_probe": data.get("false_positive_probe"),
            "cost_summary": data.get("cost_summary"),
            "golden_set": data.get("golden_set"),
            "thresholds": data.get("thresholds"),
            "source": "results.json",
        }
    partial_path = _partial_results_path(out_dir)
    if partial_path.is_file():
        cases = load_partial_results(partial_path)
        if not cases:
            raise ValueError(f"Parent run {parent_run_id} partial_results.jsonl is empty")
        cfg_path = _run_config_path(out_dir)
        cfg = load_run_config(cfg_path) if cfg_path.is_file() else {}
        return {
            "parent_run_id": str(parent_run_id),
            "path": str(partial_path),
            "cases": cases,
            "false_positive_probe": None,
            "cost_summary": None,
            "golden_set": cfg.get("gold_path"),
            "thresholds": cfg.get("thresholds_path"),
            "source": "partial_results.jsonl",
        }
    raise FileNotFoundError(
        f"No results.json or partial_results.jsonl under {out_dir}"
    )


def _new_derivative_run_id(parent_run_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_parent = re.sub(r"[^A-Za-z0-9._-]+", "_", str(parent_run_id))[:48]
    return f"{ts}_from_{safe_parent}"


def _confirm_only_failed(
    failed_rows: Sequence[dict[str, Any]],
    *,
    assume_yes: bool,
) -> bool:
    by_cat: dict[str, int] = defaultdict(int)
    for row in failed_rows:
        by_cat[str(row.get("category") or "unknown").strip().lower()] += 1
    print(f"\n--only-failed: will re-run {len(failed_rows)} case(s) with the FULL pipeline:")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")
    print()
    if assume_yes:
        print("Proceeding (--yes / EVAL_ASSUME_YES).")
        return True
    if not sys.stdin.isatty():
        print(
            "stdin is not a TTY — pass --yes to confirm --only-failed.",
            file=sys.stderr,
        )
        return False
    try:
        reply = input("Continue with full SUT+scoring re-run for these cases? [y/N] ").strip().lower()
    except EOFError:
        reply = ""
    return reply in {"y", "yes"}


def _finalize_merged_results(
    *,
    rid: str,
    out_dir: Path,
    case_results: list[dict[str, Any]],
    parent_run_id: str,
    mode: str,
    gold_path: Path | str | None,
    thresholds_path: Path | str | None,
    thresholds: dict[str, Any],
    fp_report: dict[str, Any] | None,
    cost_summary: dict[str, Any],
    estimate: dict[str, Any] | None,
    rescored_ids: Sequence[str],
    kept_ids: Sequence[str],
    merged_from_run_ids: Sequence[str] | None = None,
    gap_ids: Sequence[str] | None = None,
    n_golden: int | None = None,
) -> tuple[dict[str, Any], int]:
    """Write results.json / index / dashboard for a derivative (merged) run.

    Aggregations (per-category pass rates, RAGAS nanmean averages + coverage,
    critical-category gate, overall_status) are recomputed on the **full** merged
    case list — never on the gap subset alone.
    """
    per_category = aggregate_per_category(case_results, false_positive_probe=fp_report)
    gate = apply_thresholds(per_category, thresholds)
    ts = datetime.now(timezone.utc)
    lineage = [str(x) for x in (merged_from_run_ids or [parent_run_id, rid]) if str(x).strip()]
    # Dedupe while preserving order.
    seen_lineage: set[str] = set()
    lineage_unique: list[str] = []
    for x in lineage:
        if x not in seen_lineage:
            seen_lineage.add(x)
            lineage_unique.append(x)

    results: dict[str, Any] = {
        "run_id": rid,
        "parent_run_id": parent_run_id,
        "merged_from_run_ids": lineage_unique,
        "mode": mode,
        "timestamp": ts.isoformat(),
        "golden_set": str(gold_path or DEFAULT_GOLDEN),
        "thresholds": str(thresholds_path or DEFAULT_THRESHOLDS),
        "n_cases": len(case_results),
        "n_golden": n_golden if n_golden is not None else len(case_results),
        "n_rescored": len(list(rescored_ids)),
        "n_kept_from_parent": len(list(kept_ids)),
        "rescored_ids": list(rescored_ids),
        "kept_ids": list(kept_ids),
        "gap_ids": list(gap_ids) if gap_ids is not None else list(rescored_ids),
        "estimate": estimate,
        "cases": case_results,
        "per_category": per_category,
        "gate": {
            "overall_status": gate["overall_status"],
            "critical_failures": gate["critical_failures"],
            "noncritical_failures": gate["noncritical_failures"],
            "categories": gate["categories"],
        },
        "overall_status": gate["overall_status"],
        "cost_summary": cost_summary,
        "false_positive_probe": fp_report,
        "merge": {
            "frame": "golden_set",
            "kept_from": parent_run_id,
            "rescored_in": rid,
            "n_kept": len(list(kept_ids)),
            "n_rescored": len(list(rescored_ids)),
        },
    }
    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    latest = RESULTS_DIR / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "run_id": rid,
                "parent_run_id": parent_run_id,
                "merged_from_run_ids": lineage_unique,
                "mode": mode,
                "path": str(results_path.as_posix()),
                "overall_status": gate["overall_status"],
                "n_cases": len(case_results),
                "timestamp": ts.isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _append_results_index(
        {
            "run_id": rid,
            "parent_run_id": parent_run_id,
            "merged_from_run_ids": lineage_unique,
            "mode": mode,
            "timestamp": ts.isoformat(),
            "path": str(results_path.as_posix()),
            "overall_status": gate["overall_status"],
            "n_cases": len(case_results),
            "n_rescored": len(list(rescored_ids)),
            "total_cost_usd": cost_summary.get("total_cost_usd"),
            "total_tokens": cost_summary.get("total_tokens"),
            "wall_clock_seconds": cost_summary.get("wall_clock_seconds"),
            "critical_failures": gate["critical_failures"],
            "noncritical_failures": gate["noncritical_failures"],
            "per_category_summary": {
                cat: {
                    "pass_rate": agg.get("pass_rate"),
                    "attack_success_rate": agg.get("attack_success_rate"),
                    "n_cases": agg.get("n_cases"),
                    "ragas_averages": agg.get("ragas_averages"),
                }
                for cat, agg in per_category.items()
            },
        }
    )
    print(
        f"Wrote {results_path} (mode={mode}, n_cases={len(case_results)}, "
        f"merged_from={lineage_unique})"
    )
    for cat, info in gate["categories"].items():
        mark = "OK" if info["met_threshold"] else "FAIL"
        print(f"  [{mark}] {cat} ({info['severity']}): {info.get('failures') or 'met'}")
    print_run_cost_summary(
        n_questions=len(case_results),
        cost_summary=cost_summary,
        overall_status=gate["overall_status"],
    )
    try:
        from eval.render_dashboard import render_dashboard

        dash_path = render_dashboard(results, out_path=out_dir / "dashboard.png")
        print(f"Wrote {dash_path}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Dashboard render failed: %s", exc)
        print(f"Dashboard render failed: {exc}", file=sys.stderr)
    try:
        from eval.render_ragas_dashboard import render_ragas_dashboard

        ragas_path = render_ragas_dashboard(
            results, out_path=out_dir / "ragas_dashboard.png"
        )
        print(f"Wrote {ragas_path}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAGAS dashboard render failed: %s", exc)
        print(f"RAGAS dashboard render failed: {exc}", file=sys.stderr)
    exit_code = 1 if gate["overall_status"] == "NOT PRODUCTION READY" else 0
    return results, exit_code


def run_resume_scoring_only(
    parent_run_id: str,
    *,
    gold_path: Path | None = None,
    thresholds_path: Path | None = None,
    categories: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    run_id: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Re-score saved answers with deterministic checks only — zero LLM calls."""
    parent = load_parent_run_payload(parent_run_id)
    gold = load_golden_set(Path(gold_path or parent.get("golden_set") or DEFAULT_GOLDEN))
    by_id = {str(c.get("id")): c for c in gold}
    thr_raw = thresholds_path or parent.get("thresholds")
    thresholds = load_thresholds(Path(thr_raw) if thr_raw else DEFAULT_THRESHOLDS)
    cat_filter = set(categories) if categories else None
    id_filter = set(case_ids) if case_ids else None

    rid = run_id or _new_derivative_run_id(parent_run_id)
    out_dir = RESULTS_DIR / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = _partial_results_path(out_dir)
    if partial_path.is_file():
        partial_path.unlink()

    wall_t0 = time.perf_counter()
    merged: list[dict[str, Any]] = []
    rescored_ids: list[str] = []
    kept_ids: list[str] = []
    per_cat_usage: dict[str, dict[str, Any]] = defaultdict(_new_cat_usage_bucket)

    for saved in parent["cases"]:
        cid = str(saved.get("id") or "")
        cat = str(saved.get("category") or "").strip().lower()
        skip = False
        if cat_filter is not None and cat not in cat_filter:
            skip = True
        if id_filter is not None and cid not in id_filter:
            skip = True
        if skip:
            merged.append(dict(saved))
            kept_ids.append(cid)
            entry = saved.get("_case_cost") or _empty_category_cost_entry()
            _accumulate_case_cost(per_cat_usage[cat or "unknown"], entry)
            continue
        case = by_id.get(cid)
        if case is None:
            logger.warning("resume-scoring-only: %s missing from golden set; keeping prior row", cid)
            merged.append(dict(saved))
            kept_ids.append(cid)
            continue
        row = rescore_saved_case(case, saved)
        # Offline rescoring adds no new model spend; preserve prior case cost.
        if "_case_cost" not in row and saved.get("_case_cost"):
            row["_case_cost"] = saved["_case_cost"]
        entry = row.get("_case_cost") or _empty_category_cost_entry()
        _accumulate_case_cost(per_cat_usage[cat or "unknown"], entry)
        merged.append(row)
        rescored_ids.append(cid)
        append_partial_result(partial_path, row)

    write_run_config(
        _run_config_path(out_dir),
        {
            "run_id": rid,
            "parent_run_id": parent_run_id,
            "mode": "resume_scoring_only",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gold_path": str(gold_path or parent.get("golden_set") or DEFAULT_GOLDEN),
            "thresholds_path": str(
                thresholds_path or parent.get("thresholds") or DEFAULT_THRESHOLDS
            ),
            "categories": list(categories) if categories else None,
            "case_ids": list(case_ids) if case_ids else None,
            "rescored_ids": rescored_ids,
            "source_path": parent.get("path"),
            "llm_calls": 0,
        },
    )

    overall_usage = _overall_usage_from_per_category(per_cat_usage)
    cost_summary = build_cost_summary(
        overall=overall_usage,
        per_category={k: dict(v) for k, v in sorted(per_cat_usage.items())},
        wall_clock_seconds=time.perf_counter() - wall_t0,
    )
    cost_summary["note"] = (
        "resume-scoring-only: no new LLM calls; costs copied from parent case rows."
    )

    print(
        f"resume-scoring-only: rescored {len(rescored_ids)} case(s), "
        f"kept {len(kept_ids)} from parent {parent_run_id} "
        f"(source={parent.get('source')}); wall={cost_summary['wall_clock_seconds']:.2f}s"
    )
    return _finalize_merged_results(
        rid=rid,
        out_dir=out_dir,
        case_results=merged,
        parent_run_id=parent_run_id,
        mode="resume_scoring_only",
        gold_path=gold_path or parent.get("golden_set"),
        thresholds_path=thresholds_path or parent.get("thresholds"),
        thresholds=thresholds,
        fp_report=parent.get("false_positive_probe"),
        cost_summary=cost_summary,
        estimate={"total": 0, "note": "zero LLM calls (deterministic rescoring)"},
        rescored_ids=rescored_ids,
        kept_ids=kept_ids,
    )


def resolve_latest_run_id() -> str:
    """Resolve the most recent eval run id from ``eval/results/latest.json``."""
    latest = RESULTS_DIR / "latest.json"
    if not latest.is_file():
        raise FileNotFoundError(
            f"No {latest.as_posix()} — pass an explicit RUN_ID to --gap-set"
        )
    meta = json.loads(latest.read_text(encoding="utf-8"))
    rid = str(meta.get("run_id") or "").strip()
    if not rid:
        raise ValueError(f"{latest} has no run_id field")
    return rid


def print_gap_set_report(
    gap: dict[str, Any],
    *,
    parent_run_id: str,
    parent_source: str,
    estimate: dict[str, Any] | None = None,
) -> None:
    """Print gap-set size, category breakdown, and optional LLM-call estimate."""
    print(f"\n=== Gap set vs golden_set (parent={parent_run_id}, source={parent_source}) ===")
    print(
        f"Golden cases: {gap['n_golden']}  ·  Parent result rows: {gap['n_results']}  ·  "
        f"Passed: {gap['n_passed']}  ·  Gap: {gap['n_gap']}"
    )
    print(
        f"  missing (never attempted): {gap['n_missing']}  ·  "
        f"error (pass=false + error): {gap['n_error']}  ·  "
        f"failed (scored pass=false): {gap['n_failed']}"
    )
    print("\nBy category:")
    reasons = gap.get("by_category_reason") or {}
    for cat, n in (gap.get("by_category") or {}).items():
        detail = reasons.get(cat) or {}
        parts = [
            f"{k}={detail[k]}"
            for k in ("missing", "error", "failed")
            if int(detail.get(k) or 0) > 0
        ]
        suffix = f" ({', '.join(parts)})" if parts else ""
        print(f"  {cat}: {n}{suffix}")
    if gap.get("missing_ids"):
        print(f"\nMissing ids ({len(gap['missing_ids'])}): {', '.join(gap['missing_ids'])}")
    if gap.get("error_ids"):
        print(f"Error ids ({len(gap['error_ids'])}): {', '.join(gap['error_ids'])}")
    if estimate is not None:
        print(
            f"\nEstimated LLM calls for gap subset: {estimate.get('total')} "
            f"(system={estimate.get('system_calls')}, "
            f"judge={estimate.get('judge_calls')}, "
            f"guard={estimate.get('guard_calls')})"
        )
        if estimate.get("note"):
            print(estimate["note"])
    print()


def _confirm_gap_set(
    gap: dict[str, Any],
    *,
    assume_yes: bool,
    flag_name: str = "--fill-gaps",
) -> bool:
    n = int(gap.get("n_gap") or 0)
    print(
        f"{flag_name}: will re-run {n} case(s) with the FULL pipeline "
        "(SUT answer + custom checks + RAGAS + DeepEval/DeepTeam; missing + failed + error)."
    )
    if assume_yes:
        print("Proceeding (--yes / EVAL_ASSUME_YES).")
        return True
    if not sys.stdin.isatty():
        print(
            f"stdin is not a TTY — pass --yes to confirm {flag_name}.",
            file=sys.stderr,
        )
        return False
    try:
        reply = input(
            "Continue with full SUT+scoring for the gap set? [y/N] "
        ).strip().lower()
    except EOFError:
        reply = ""
    return reply in {"y", "yes"}


def run_gap_set(
    parent_run_id: str | None = None,
    *,
    gold_path: Path | None = None,
    thresholds_path: Path | None = None,
    llm: LLMClient | None = None,
    assume_yes: bool = False,
    skip_ragas: bool = False,
    categories: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    run_id: str | None = None,
    skip_preflight: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    report_only: bool = False,
    resume_run_id: str | None = None,
    flag_name: str = "--fill-gaps",
) -> tuple[dict[str, Any], int]:
    """Full SUT+scoring for golden cases missing or failing in the parent run.

    Gap membership = no result row **or** ``pass`` is not true (includes
    unrecoverable ``error`` rows). Writes a NEW derivative run dir.

    Checkpointing: each scored gap case is appended to
    ``partial_results.jsonl`` immediately. Interrupt and continue with
    ``python -m eval.run_full --resume {derivative_run_id}``.
    """
    assume = assume_yes or (os.getenv("EVAL_ASSUME_YES") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    resuming = bool(resume_run_id)
    run_config: dict[str, Any] = {}

    if resuming:
        rid = str(resume_run_id)
        out_dir = RESULTS_DIR / rid
        config_path = _run_config_path(out_dir)
        if not config_path.is_file():
            print(
                f"Aborted: no run_config.json for --resume {rid!r} under {out_dir}.",
                file=sys.stderr,
            )
            return {"aborted": True, "reason": f"missing run_config for {rid}"}, 2
        run_config = load_run_config(config_path)
        mode = str(run_config.get("mode") or "")
        if mode not in {"gap_set", "fill_gaps"}:
            print(
                f"Aborted: run {rid} mode={mode!r} is not a fill-gaps/gap-set run.",
                file=sys.stderr,
            )
            return {"aborted": True, "reason": "resume_not_fill_gaps"}, 2
        rid_parent = str(run_config.get("parent_run_id") or parent_run_id or "")
        if not rid_parent:
            print(
                f"Aborted: resume {rid} is missing parent_run_id in run_config.json.",
                file=sys.stderr,
            )
            return {"aborted": True, "reason": "missing_parent_run_id"}, 2
        gold_path = Path(run_config["gold_path"]) if run_config.get("gold_path") else gold_path
        thresholds_path = (
            Path(run_config["thresholds_path"])
            if run_config.get("thresholds_path")
            else thresholds_path
        )
        skip_ragas = bool(run_config.get("skip_ragas", skip_ragas))
        batch_size = int(run_config.get("batch_size") or batch_size)
        categories = run_config.get("categories") or categories
        # Gap membership is frozen in the original fill-gaps run_config.
        frozen_gap_ids = [
            str(cid) for cid in (run_config.get("gap_ids") or run_config.get("case_ids") or [])
        ]
    else:
        rid_parent = parent_run_id or resolve_latest_run_id()
        frozen_gap_ids = []

    parent = load_parent_run_payload(rid_parent)
    gold = load_golden_set(Path(gold_path or parent.get("golden_set") or DEFAULT_GOLDEN))
    by_id_gold = {str(c.get("id")): c for c in gold}

    if resuming and frozen_gap_ids:
        gap_cases = [by_id_gold[cid] for cid in frozen_gap_ids if cid in by_id_gold]
        # Reconstruct a minimal gap report for the console / estimate.
        gap = compute_gap_set(gold, parent["cases"])
        gap_id_set_frozen = set(frozen_gap_ids)
        filtered_rows = [r for r in gap["gap_rows"] if r["id"] in gap_id_set_frozen]
        # Include frozen ids even if parent now somehow passes (still finish the fill).
        present = {r["id"] for r in filtered_rows}
        for cid in frozen_gap_ids:
            if cid in present or cid not in by_id_gold:
                continue
            case = by_id_gold[cid]
            filtered_rows.append(
                {
                    "id": cid,
                    "category": str(case.get("category") or "unknown").strip().lower(),
                    "severity": case.get("severity"),
                    "reason": "failed",
                    "pass": False,
                    "has_result": True,
                    "error": None,
                }
            )
        by_cat: dict[str, int] = defaultdict(int)
        by_cat_reason: dict[str, dict[str, int]] = defaultdict(
            lambda: {"missing": 0, "error": 0, "failed": 0}
        )
        for row in filtered_rows:
            by_cat[row["category"]] += 1
            by_cat_reason[row["category"]][row.get("reason") or "failed"] += 1
        gap = {
            **gap,
            "gap_rows": filtered_rows,
            "gap_ids": list(frozen_gap_ids),
            "gap_cases": gap_cases,
            "n_gap": len(gap_cases),
            "by_category": dict(sorted(by_cat.items())),
            "by_category_reason": {
                cat: dict(reasons) for cat, reasons in sorted(by_cat_reason.items())
            },
            "n_missing": sum(1 for r in filtered_rows if r.get("reason") == "missing"),
            "n_error": sum(1 for r in filtered_rows if r.get("reason") == "error"),
            "n_failed": sum(1 for r in filtered_rows if r.get("reason") == "failed"),
            "missing_ids": [r["id"] for r in filtered_rows if r.get("reason") == "missing"],
            "error_ids": [r["id"] for r in filtered_rows if r.get("reason") == "error"],
            "failed_ids": [r["id"] for r in filtered_rows if r.get("reason") == "failed"],
        }
    else:
        gap = compute_gap_set(gold, parent["cases"])
        cat_filter = set(categories) if categories else None
        id_filter = set(case_ids) if case_ids else None
        if cat_filter or id_filter:
            filtered_rows = []
            for row in gap["gap_rows"]:
                if cat_filter is not None and row["category"] not in cat_filter:
                    continue
                if id_filter is not None and row["id"] not in id_filter:
                    continue
                filtered_rows.append(row)
            gap_ids = [r["id"] for r in filtered_rows]
            by_cat = defaultdict(int)
            by_cat_reason = defaultdict(lambda: {"missing": 0, "error": 0, "failed": 0})
            for row in filtered_rows:
                by_cat[row["category"]] += 1
                by_cat_reason[row["category"]][row["reason"]] += 1
            gap = {
                **gap,
                "gap_rows": filtered_rows,
                "gap_ids": gap_ids,
                "gap_cases": [by_id_gold[cid] for cid in gap_ids if cid in by_id_gold],
                "n_gap": len(gap_ids),
                "by_category": dict(sorted(by_cat.items())),
                "by_category_reason": {
                    cat: dict(reasons) for cat, reasons in sorted(by_cat_reason.items())
                },
                "n_missing": sum(1 for r in filtered_rows if r["reason"] == "missing"),
                "n_error": sum(1 for r in filtered_rows if r["reason"] == "error"),
                "n_failed": sum(1 for r in filtered_rows if r["reason"] == "failed"),
                "missing_ids": [r["id"] for r in filtered_rows if r["reason"] == "missing"],
                "error_ids": [r["id"] for r in filtered_rows if r["reason"] == "error"],
                "failed_ids": [r["id"] for r in filtered_rows if r["reason"] == "failed"],
            }
        gap_cases = list(gap["gap_cases"])

    rewrite_enabled = (os.getenv("RETRIEVAL_REWRITE") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    # On resume, only estimate remaining (not-yet-checkpointed) cases.
    partial_path_preview = (
        _partial_results_path(RESULTS_DIR / str(resume_run_id))
        if resuming
        else None
    )
    already_done_ids: set[str] = set()
    if partial_path_preview and partial_path_preview.is_file():
        already_done_ids = {
            str(r.get("id") or "")
            for r in load_partial_results(partial_path_preview)
            if r.get("id")
        }
    remaining_cases = [
        c for c in gap_cases if str(c.get("id") or "") not in already_done_ids
    ]
    estimate = estimate_run_llm_calls(
        remaining_cases if resuming else gap_cases,
        rewrite_enabled=rewrite_enabled,
        skip_ragas=skip_ragas,
        include_security_fp=False,
    )
    print_gap_set_report(
        gap,
        parent_run_id=rid_parent,
        parent_source=str(parent.get("source") or ""),
        estimate=estimate,
    )
    if resuming:
        print(
            f"Resuming fill-gaps run {resume_run_id}: "
            f"{len(already_done_ids)} checkpointed, {len(remaining_cases)} remaining.\n"
        )

    if report_only:
        return {"gap": gap, "estimate": estimate, "aborted": True, "reason": "report_only"}, 0

    if not gap_cases:
        print("Gap set is empty — nothing to re-run.")
        return {"gap": gap, "aborted": True, "reason": "empty_gap"}, 0

    if not resuming and not _confirm_gap_set(gap, assume_yes=assume, flag_name=flag_name):
        print(f"Aborted ({flag_name} not confirmed).", file=sys.stderr)
        return {"gap": gap, "aborted": True, "reason": "gap_set_declined"}, 2

    if not skip_preflight:
        print("Preflight: pinging each Portkey provider directly (bypassing fallback)...\n")
        preflight_results = run_preflight()
        print(_format_preflight_table(preflight_results))
        if not _confirm_preflight(preflight_results, assume_yes=assume):
            print("Aborted (preflight).", file=sys.stderr)
            return {"aborted": True, "preflight": preflight_results, "gap": gap}, 2
        print()

    thr_raw = thresholds_path or parent.get("thresholds")
    thresholds = load_thresholds(Path(thr_raw) if thr_raw else DEFAULT_THRESHOLDS)
    budget = (thresholds.get("call_budget") or {}).get("max_estimated_calls", 400)
    try:
        budget = int(os.getenv("EVAL_MAX_ESTIMATED_CALLS") or budget)
    except ValueError:
        budget = int(budget)
    if not _confirm_call_budget(estimate, max_estimated_calls=budget, assume_yes=assume):
        print("Aborted.", file=sys.stderr)
        return {"aborted": True, "estimate": estimate, "gap": gap}, 2

    client = llm or LLMClient()
    if resuming:
        rid = str(resume_run_id)
        out_dir = RESULTS_DIR / rid
    else:
        rid = run_id or _new_derivative_run_id(rid_parent)
        out_dir = RESULTS_DIR / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        partial_path = _partial_results_path(out_dir)
        if partial_path.is_file():
            partial_path.unlink()
        write_run_config(
            _run_config_path(out_dir),
            {
                "run_id": rid,
                "parent_run_id": rid_parent,
                "mode": "fill_gaps",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "gold_path": str(gold_path or parent.get("golden_set") or DEFAULT_GOLDEN),
                "thresholds_path": str(
                    thresholds_path or parent.get("thresholds") or DEFAULT_THRESHOLDS
                ),
                "categories": list(categories) if categories else None,
                # case_ids frozen for --resume (same contract as full runs).
                "case_ids": list(gap["gap_ids"]),
                "skip_ragas": skip_ragas,
                "batch_size": batch_size,
                "gap_ids": list(gap["gap_ids"]),
                "missing_ids": list(gap["missing_ids"]),
                "error_ids": list(gap["error_ids"]),
                "failed_ids": list(gap["failed_ids"]),
            },
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = _partial_results_path(out_dir)

    log_path = Path(client.log_path)
    wall_t0 = time.perf_counter()
    security_judge = None
    guard_in = guard_out = None
    need_security = any(
        str(c.get("category") or "").strip().lower() in SECURITY_CATEGORIES
        for c in gap_cases
    )
    if need_security:
        from eval.scoring.security_scorer import (
            _build_topic_guards,
            _portkey_security_judge,
        )

        security_judge = _portkey_security_judge(client)
        guard_in, guard_out = _build_topic_guards(security_judge)

    per_cat_usage: dict[str, dict[str, Any]] = defaultdict(_new_cat_usage_bucket)
    gap_id_set = {str(c.get("id")) for c in gap_cases}
    kept_by_id: dict[str, dict[str, Any]] = {}
    for saved in parent["cases"]:
        cid = str(saved.get("id") or "")
        if cid in gap_id_set:
            continue
        if not saved.get("pass"):
            continue
        kept_by_id[cid] = dict(saved)
        cat = str(saved.get("category") or "unknown").strip().lower()
        _accumulate_case_cost(
            per_cat_usage[cat], saved.get("_case_cost") or _empty_category_cost_entry()
        )

    # Seed from checkpoints so an interrupted fill-gaps run can continue.
    rescored_by_id: dict[str, dict[str, Any]] = {}
    for saved in load_partial_results(partial_path):
        cid = str(saved.get("id") or "")
        if not cid or cid not in gap_id_set:
            continue
        rescored_by_id[cid] = dict(saved)
        cat = str(saved.get("category") or "unknown").strip().lower()
        _accumulate_case_cost(
            per_cat_usage[cat], saved.get("_case_cost") or _empty_category_cost_entry()
        )

    todo_cases = [c for c in gap_cases if str(c.get("id") or "") not in rescored_by_id]
    batches = chunked(todo_cases, batch_size)
    n_total_batches = len(batches)
    for batch_idx, batch in enumerate(batches, 1):
        for case in batch:
            cat = str(case.get("category") or "").strip().lower() or "unknown"
            case_t0 = time.perf_counter()
            case_log_start = _usage_snapshot(log_path)
            try:
                row = _score_case_row(
                    case,
                    llm=client,
                    skip_ragas=skip_ragas,
                    security_judge=security_judge,
                    guard_input=guard_in,
                    guard_output=guard_out,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("case %s failed", case.get("id"))
                row = {
                    "id": case.get("id"),
                    "category": cat,
                    "severity": case.get("severity"),
                    "pass": False,
                    "error": str(exc),
                }
            case_log_end = _usage_snapshot(log_path)
            slice_usage = summarize_usage_since(
                log_path, start_size=case_log_start, end_size=case_log_end
            )
            entry = _category_cost_entry(slice_usage)
            row["_case_cost"] = entry
            row["_case_timing_seconds"] = round(time.perf_counter() - case_t0, 3)
            row["_batch_index"] = batch_idx
            row["parent_run_id"] = rid_parent
            _accumulate_case_cost(per_cat_usage[cat], entry)
            rescored_by_id[str(case.get("id"))] = row
            append_partial_result(partial_path, row)
        print(
            f"fill-gaps batch {batch_idx}/{max(n_total_batches, 1)} complete "
            f"({len(rescored_by_id)}/{len(gap_cases)} gap cases done)."
        )
        if batch_idx < n_total_batches:
            running = _overall_usage_from_per_category(per_cat_usage)
            if not check_budget_ceiling(
                total_tokens=running["total_tokens"],
                total_cost_usd=running["total_cost_usd"],
                thresholds=thresholds,
                assume_yes=assume,
                batch_label=f"fill-gaps batch {batch_idx + 1}/{n_total_batches}",
            ):
                print(
                    f"Paused after fill-gaps batch {batch_idx}/{n_total_batches} "
                    f"({len(rescored_by_id)}/{len(gap_cases)} gap cases). Resume with:\n"
                    f"  python -m eval.run_full --resume {rid}",
                    file=sys.stderr,
                )
                return {
                    "paused": True,
                    "run_id": rid,
                    "parent_run_id": rid_parent,
                    "mode": "fill_gaps",
                    "batches_completed": batch_idx,
                    "cases_completed": len(rescored_by_id),
                    "gap": gap,
                }, 3

    # --- Merge onto the full golden-set frame (complete picture, not gap subset) ---
    try:
        merged_payload = merge_fill_gaps_results(
            gold,
            parent_cases=parent["cases"],
            gap_ids=list(gap["gap_ids"]),
            new_gap_results=rescored_by_id,
        )
    except GapMergeError as exc:
        print(f"FATAL: fill-gaps merge sanity check failed: {exc}", file=sys.stderr)
        logger.exception("fill-gaps merge failed")
        return {
            "aborted": True,
            "reason": "merge_sanity_failed",
            "error": str(exc),
            "run_id": rid,
            "parent_run_id": rid_parent,
            "gap": gap,
            "n_rescored": len(rescored_by_id),
        }, 2

    print(
        f"Merge OK: {merged_payload['n_merged']}/{merged_payload['n_golden']} golden cases "
        f"(kept={len(merged_payload['kept_ids'])}, "
        f"rescored={len(merged_payload['rescored_ids'])})."
    )

    overall_usage = _overall_usage_from_per_category(per_cat_usage)
    cost_summary = build_cost_summary(
        overall=overall_usage,
        per_category={k: dict(v) for k, v in sorted(per_cat_usage.items())},
        wall_clock_seconds=time.perf_counter() - wall_t0,
    )
    return _finalize_merged_results(
        rid=rid,
        out_dir=out_dir,
        case_results=merged_payload["cases"],
        parent_run_id=rid_parent,
        mode="fill_gaps",
        gold_path=gold_path or parent.get("golden_set"),
        thresholds_path=thresholds_path or parent.get("thresholds"),
        thresholds=thresholds,
        fp_report=parent.get("false_positive_probe"),
        cost_summary=cost_summary,
        estimate=estimate,
        rescored_ids=merged_payload["rescored_ids"],
        kept_ids=merged_payload["kept_ids"],
        merged_from_run_ids=[rid_parent, rid],
        gap_ids=merged_payload["gap_ids"],
        n_golden=merged_payload["n_golden"],
    )


def run_only_failed(
    parent_run_id: str,
    *,
    gold_path: Path | None = None,
    thresholds_path: Path | None = None,
    llm: LLMClient | None = None,
    assume_yes: bool = False,
    skip_ragas: bool = False,
    include_security_fp: bool = False,
    categories: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    run_id: str | None = None,
    skip_preflight: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[dict[str, Any], int]:
    """Full SUT+scoring re-run for parent cases with pass=False; merge with keepers."""
    assume = assume_yes or (os.getenv("EVAL_ASSUME_YES") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    parent = load_parent_run_payload(parent_run_id)
    cat_filter = set(categories) if categories else None
    id_filter = set(case_ids) if case_ids else None
    failed = [
        r
        for r in parent["cases"]
        if not r.get("pass")
        and (
            cat_filter is None
            or str(r.get("category") or "").strip().lower() in cat_filter
        )
        and (id_filter is None or str(r.get("id") or "") in id_filter)
    ]
    if not failed:
        print(f"No failed cases to re-run in parent {parent_run_id}.")
        return {"aborted": True, "reason": "no_failed_cases"}, 0
    if not _confirm_only_failed(failed, assume_yes=assume):
        print("Aborted (--only-failed not confirmed).", file=sys.stderr)
        return {"aborted": True, "reason": "only_failed_declined"}, 2

    if not skip_preflight:
        print("Preflight: pinging each Portkey provider directly (bypassing fallback)...\n")
        preflight_results = run_preflight()
        print(_format_preflight_table(preflight_results))
        if not _confirm_preflight(preflight_results, assume_yes=assume):
            print("Aborted (preflight).", file=sys.stderr)
            return {"aborted": True, "preflight": preflight_results}, 2
        print()

    gold = load_golden_set(Path(gold_path or parent.get("golden_set") or DEFAULT_GOLDEN))
    by_id = {str(c.get("id")): c for c in gold}
    failed_cases: list[dict[str, Any]] = []
    for row in failed:
        cid = str(row.get("id") or "")
        case = by_id.get(cid)
        if case is None:
            logger.warning("only-failed: %s missing from golden set; skipping", cid)
            continue
        failed_cases.append(case)

    thr_raw = thresholds_path or parent.get("thresholds")
    thresholds = load_thresholds(Path(thr_raw) if thr_raw else DEFAULT_THRESHOLDS)
    client = llm or LLMClient()
    rid = run_id or _new_derivative_run_id(parent_run_id)
    out_dir = RESULTS_DIR / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = _partial_results_path(out_dir)
    if partial_path.is_file():
        partial_path.unlink()

    write_run_config(
        _run_config_path(out_dir),
        {
            "run_id": rid,
            "parent_run_id": parent_run_id,
            "mode": "only_failed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gold_path": str(gold_path or parent.get("golden_set") or DEFAULT_GOLDEN),
            "thresholds_path": str(
                thresholds_path or parent.get("thresholds") or DEFAULT_THRESHOLDS
            ),
            "categories": list(categories) if categories else None,
            "case_ids": list(case_ids) if case_ids else None,
            "skip_ragas": skip_ragas,
            "batch_size": batch_size,
            "case_ids_rerun": [c.get("id") for c in failed_cases],
            "failed_ids_from_parent": [r.get("id") for r in failed],
        },
    )

    rewrite_enabled = (os.getenv("RETRIEVAL_REWRITE") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    estimate = estimate_run_llm_calls(
        failed_cases,
        rewrite_enabled=rewrite_enabled,
        skip_ragas=skip_ragas,
        include_security_fp=False,
    )
    budget = (thresholds.get("call_budget") or {}).get("max_estimated_calls", 400)
    try:
        budget = int(os.getenv("EVAL_MAX_ESTIMATED_CALLS") or budget)
    except ValueError:
        budget = int(budget)
    if not _confirm_call_budget(estimate, max_estimated_calls=budget, assume_yes=assume):
        print("Aborted.", file=sys.stderr)
        return {"aborted": True, "estimate": estimate}, 2

    log_path = Path(client.log_path)
    wall_t0 = time.perf_counter()
    security_judge = None
    guard_in = guard_out = None
    need_security = any(
        str(c.get("category") or "").strip().lower() in SECURITY_CATEGORIES
        for c in failed_cases
    )
    if need_security:
        from eval.scoring.security_scorer import (
            _build_topic_guards,
            _portkey_security_judge,
        )

        security_judge = _portkey_security_judge(client)
        guard_in, guard_out = _build_topic_guards(security_judge)

    per_cat_usage: dict[str, dict[str, Any]] = defaultdict(_new_cat_usage_bucket)
    failed_id_set = {str(c.get("id")) for c in failed_cases}
    kept_by_id: dict[str, dict[str, Any]] = {}
    for saved in parent["cases"]:
        cid = str(saved.get("id") or "")
        if cid in failed_id_set:
            continue
        kept_by_id[cid] = dict(saved)
        cat = str(saved.get("category") or "unknown").strip().lower()
        _accumulate_case_cost(
            per_cat_usage[cat], saved.get("_case_cost") or _empty_category_cost_entry()
        )

    rescored_by_id: dict[str, dict[str, Any]] = {}
    batches = chunked(failed_cases, batch_size)
    for batch_idx, batch in enumerate(batches, 1):
        for case in batch:
            cat = str(case.get("category") or "").strip().lower() or "unknown"
            case_t0 = time.perf_counter()
            case_log_start = _usage_snapshot(log_path)
            try:
                row = _score_case_row(
                    case,
                    llm=client,
                    skip_ragas=skip_ragas,
                    security_judge=security_judge,
                    guard_input=guard_in,
                    guard_output=guard_out,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("case %s failed", case.get("id"))
                row = {
                    "id": case.get("id"),
                    "category": cat,
                    "severity": case.get("severity"),
                    "pass": False,
                    "error": str(exc),
                }
            case_log_end = _usage_snapshot(log_path)
            slice_usage = summarize_usage_since(
                log_path, start_size=case_log_start, end_size=case_log_end
            )
            entry = _category_cost_entry(slice_usage)
            row["_case_cost"] = entry
            row["_case_timing_seconds"] = round(time.perf_counter() - case_t0, 3)
            row["_batch_index"] = batch_idx
            row["parent_run_id"] = parent_run_id
            _accumulate_case_cost(per_cat_usage[cat], entry)
            rescored_by_id[str(case.get("id"))] = row
            append_partial_result(partial_path, row)
        print(
            f"only-failed batch {batch_idx}/{len(batches)} complete "
            f"({len(rescored_by_id)}/{len(failed_cases)} re-run)."
        )

    # Preserve parent order; swap in re-run rows.
    merged: list[dict[str, Any]] = []
    for saved in parent["cases"]:
        cid = str(saved.get("id") or "")
        if cid in rescored_by_id:
            merged.append(rescored_by_id[cid])
        elif cid in kept_by_id:
            merged.append(kept_by_id[cid])
        else:
            merged.append(dict(saved))

    overall_usage = _overall_usage_from_per_category(per_cat_usage)
    cost_summary = build_cost_summary(
        overall=overall_usage,
        per_category={k: dict(v) for k, v in sorted(per_cat_usage.items())},
        wall_clock_seconds=time.perf_counter() - wall_t0,
    )
    return _finalize_merged_results(
        rid=rid,
        out_dir=out_dir,
        case_results=merged,
        parent_run_id=parent_run_id,
        mode="only_failed",
        gold_path=gold_path or parent.get("golden_set"),
        thresholds_path=thresholds_path or parent.get("thresholds"),
        thresholds=thresholds,
        fp_report=parent.get("false_positive_probe"),
        cost_summary=cost_summary,
        estimate=estimate,
        rescored_ids=list(rescored_by_id.keys()),
        kept_ids=list(kept_by_id.keys()),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full golden-set eval (RAGAS + custom gates + security) with per-category gates"
    )
    p.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Path to golden_set.jsonl (default: eval/golden_set.jsonl, 30-case canonical set)",
    )
    p.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Path to thresholds.yaml (default: eval/thresholds.yaml)",
    )
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation when estimated calls exceed safety threshold",
    )
    p.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS LLM-judge metrics (substring + hard gates still run)",
    )
    p.add_argument(
        "--skip-security-fp",
        action="store_true",
        help="Skip guardrail false-positive probe on legitimate questions",
    )
    p.add_argument("--limit", type=int, default=None, help="Score only the first N cases")
    p.add_argument(
        "--category",
        action="append",
        default=None,
        metavar="NAME",
        help="Restrict to category name(s); repeatable and/or comma-separated "
        "(e.g. --category cross_regulation,design_implication)",
    )
    p.add_argument("--run-id", default=None, help="Override results subdirectory name")
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the direct per-provider ping check (not recommended)",
    )
    smoke_group = p.add_mutually_exclusive_group()
    smoke_group.add_argument(
        "--smoke-first",
        action="store_true",
        help=(
            "Run the 5-case smoke subset first, report results, and confirm before "
            "the full run (default behavior — this flag just makes it explicit)"
        ),
    )
    smoke_group.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the 5-case smoke subset (full pipeline, all scoring layers) and exit",
    )
    smoke_group.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the smoke subset entirely and go straight to the full run (not recommended)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Cases per batch (default {DEFAULT_BATCH_SIZE}); persisted to "
        "partial_results.jsonl after each case",
    )
    p.add_argument(
        "--resume",
        dest="resume_run_id",
        default=None,
        metavar="RUN_ID",
        help="Resume an interrupted run: read eval/results/RUN_ID/partial_results.jsonl, "
        "skip already-completed case ids, and continue",
    )
    p.add_argument(
        "--resume-scoring-only",
        dest="resume_scoring_only",
        default=None,
        metavar="RUN_ID",
        help="Re-run deterministic scoring (substring, custom_checks, injection ASR) on "
        "saved answers from RUN_ID — zero SUT/judge/guard LLM calls. Writes a NEW run "
        "dir referencing parent_run_id.",
    )
    p.add_argument(
        "--only-failed",
        dest="only_failed",
        default=None,
        metavar="RUN_ID",
        help="Re-run the FULL pipeline for cases with pass=False in RUN_ID; keep passing "
        "rows. Prints counts/categories and requires confirmation. Writes a NEW run dir.",
    )
    p.add_argument(
        "--fill-gaps",
        dest="fill_gaps",
        default=None,
        metavar="RUN_ID",
        help="Compute the gap set vs golden_set.jsonl for RUN_ID (missing + pass=False, "
        "including unrecoverable error rows) and re-run the FULL pipeline for ONLY those "
        "cases. Batched with partial_results.jsonl checkpointing; resume with --resume. "
        "Preflight + call-budget checks apply. Writes a NEW run dir.",
    )
    p.add_argument(
        "--gap-set",
        dest="gap_set",
        nargs="?",
        const="",
        default=None,
        metavar="RUN_ID",
        help="Alias for --fill-gaps. Omit RUN_ID to use eval/results/latest.json.",
    )
    p.add_argument(
        "--gap-set-report-only",
        action="store_true",
        help="With --fill-gaps / --gap-set: print the gap report + estimate and exit "
        "(no LLM calls)",
    )
    p.add_argument(
        "--case-id",
        action="append",
        default=None,
        metavar="ID",
        help="Restrict to case id(s); repeatable and/or comma-separated "
        "(useful with --only-failed / --resume-scoring-only / --fill-gaps)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        return _main_impl(argv)
    finally:
        shutdown_case_timeout_pool()


def _main_impl(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    categories = normalize_category_filters(args.category)
    case_ids = normalize_case_id_filters(args.case_id)

    exclusive = [
        bool(args.resume_run_id),
        bool(args.resume_scoring_only),
        bool(args.only_failed),
        bool(args.fill_gaps),
        args.gap_set is not None,
    ]
    if sum(exclusive) > 1:
        print(
            "Choose only one of --resume, --resume-scoring-only, --only-failed, "
            "--fill-gaps / --gap-set.",
            file=sys.stderr,
        )
        return 2

    if args.gap_set_report_only and not args.fill_gaps and args.gap_set is None:
        print(
            "--gap-set-report-only requires --fill-gaps RUN_ID or --gap-set [RUN_ID].",
            file=sys.stderr,
        )
        return 2

    if args.resume_run_id:
        cfg_path = RESULTS_DIR / str(args.resume_run_id) / "run_config.json"
        if cfg_path.is_file():
            cfg = load_run_config(cfg_path)
            mode = str(cfg.get("mode") or "")
            if mode in {"fill_gaps", "gap_set"}:
                _, code = run_gap_set(
                    resume_run_id=args.resume_run_id,
                    gold_path=args.gold,
                    thresholds_path=args.thresholds,
                    assume_yes=args.yes,
                    skip_ragas=args.skip_ragas,
                    skip_preflight=args.skip_preflight,
                    batch_size=args.batch_size,
                    flag_name="--fill-gaps",
                )
                return code

    if args.resume_scoring_only:
        _, code = run_resume_scoring_only(
            args.resume_scoring_only,
            gold_path=args.gold,
            thresholds_path=args.thresholds,
            categories=categories,
            case_ids=case_ids,
            run_id=args.run_id,
        )
        return code

    if args.only_failed:
        _, code = run_only_failed(
            args.only_failed,
            gold_path=args.gold,
            thresholds_path=args.thresholds,
            assume_yes=args.yes,
            skip_ragas=args.skip_ragas,
            include_security_fp=not args.skip_security_fp,
            categories=categories,
            case_ids=case_ids,
            run_id=args.run_id,
            skip_preflight=args.skip_preflight,
            batch_size=args.batch_size,
        )
        return code

    if args.fill_gaps or args.gap_set is not None:
        if args.fill_gaps:
            parent = str(args.fill_gaps).strip() or None
            flag_name = "--fill-gaps"
        else:
            parent = (args.gap_set or "").strip() or None
            flag_name = "--gap-set"
        if not parent and flag_name == "--fill-gaps":
            print("--fill-gaps requires RUN_ID (previous run to fill gaps from).", file=sys.stderr)
            return 2
        _, code = run_gap_set(
            parent,
            gold_path=args.gold,
            thresholds_path=args.thresholds,
            assume_yes=args.yes,
            skip_ragas=args.skip_ragas,
            categories=categories,
            case_ids=case_ids,
            run_id=args.run_id,
            skip_preflight=args.skip_preflight,
            batch_size=args.batch_size,
            report_only=args.gap_set_report_only,
            flag_name=flag_name,
        )
        return code

    _, code = run_full_eval(
        gold_path=args.gold,
        thresholds_path=args.thresholds,
        assume_yes=args.yes,
        skip_ragas=args.skip_ragas,
        include_security_fp=not args.skip_security_fp,
        limit=args.limit,
        categories=categories,
        case_ids=case_ids,
        run_id=args.run_id,
        skip_preflight=args.skip_preflight,
        smoke_only=args.smoke_only,
        skip_smoke=args.skip_smoke,
        batch_size=args.batch_size,
        resume_run_id=args.resume_run_id,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
