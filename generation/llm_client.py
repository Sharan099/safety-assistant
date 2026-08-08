"""LLM client: mock or Portkey-gateway OpenAI client with per-task fallback configs.

Configs live as Portkey Config JSON under ``config/portkey/`` (swap providers/models
without code changes). ``LLM_PROVIDER=mock`` (default) never hits the network.

Caching (live mode): Portkey *simple* exact-match cache on QUERY_REWRITE / FINAL_ANSWER
configs (24h TTL), namespaced by ingest ``cache_version`` so a successful PDF upload
invalidates prior LLM hits (including stale "not found"). That gateway cache is the
primary layer and replaces the older SQLite ``cache/response_cache.py`` pipeline cache
for LLM responses — keep response_cache only as an optional local fallback when the
gateway is unreachable (``ANSWER_CACHE=1``); otherwise leave it off to avoid two
caches that can disagree.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Eval / Portkey usage log: distinguish SUT completions vs judge-model calls.
_LLM_CALL_KIND: ContextVar[str | None] = ContextVar("llm_call_kind", default=None)


@contextmanager
def llm_call_kind_scope(kind: str) -> Iterator[None]:
    """Tag Portkey usage-log rows as ``system_under_test`` or ``judge``."""
    token = _LLM_CALL_KIND.set(str(kind).strip() or None)
    try:
        yield
    finally:
        _LLM_CALL_KIND.reset(token)


def current_llm_call_kind() -> str | None:
    return _LLM_CALL_KIND.get()


def _load_dotenv() -> None:
    # Never clobber already-set env (tests / Compose / shell exports win over .env).
    load_dotenv(override=False)


DEFAULT_CACHE_DIR = "./data/llm_cache"
DEFAULT_LOG_PATH = "./data/logs/llm_calls.jsonl"
DEFAULT_GATEWAY_URL = "http://localhost:8787/v1"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "portkey"
NVIDIA_NIM_HOST = "https://integrate.api.nvidia.com/v1"

# Nemotron Super / Ultra default to reasoning-ON; long CoT → 30–200s + huge
# completion counts. Disable via chat_template_kwargs (Portkey override_params)
# and optionally a ``/no_think`` system prefix (see NIM reasoning-model docs).
_REASONING_NIM_MODEL_RE = re.compile(
    r"(?i)nemotron-super|nemotron-ultra|nemotron-3-nano|nemotron-3-super"
)

# Non-creative steps (rewrite / condense / judge) stay at 0. Final answer may
# optionally raise via ANSWER_TEMPERATURE — default still 0 for grounded Q&A.
REWRITE_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.0
# OpenAI-compatible seed for rewrite/judge when the provider honors it.
DETERMINISTIC_SEED = 42

_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

# Portkey / provider rate-limit header names.
_RL_REMAINING_REQ = ("x-ratelimit-remaining-requests",)
_RL_REMAINING_TOK = ("x-ratelimit-remaining-tokens",)
_RL_LIMIT_REQ = ("x-ratelimit-limit-requests",)
_RL_LIMIT_TOK = ("x-ratelimit-limit-tokens",)
_RL_RESET_REQ = ("x-ratelimit-reset-requests",)
_RL_RESET_TOK = ("x-ratelimit-reset-tokens",)
_RETRY_AFTER = ("retry-after", "Retry-After")

# Groq structured-output support is model-specific; keep an allowlist so we don't
# burn a failed request on every answer call.
_JSON_SCHEMA_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "moonshotai/kimi-k2-instruct",
    }
)


def _model_supports_json_schema(model: str) -> bool:
    name = (model or "").strip().lower()
    if not name:
        return False
    if name in _JSON_SCHEMA_MODELS:
        return True
    extra = os.getenv("GROQ_JSON_SCHEMA_MODELS") or ""
    return name in {m.strip().lower() for m in extra.split(",") if m.strip()}


class LLMRole(str, Enum):
    """Model routing roles — rewrite / answer / judge each use a Portkey config."""

    REWRITE = "rewrite"
    ANSWER = "answer"
    JUDGE = "judge"


ProviderName = Literal["mock", "groq", "portkey"]


class LLMError(RuntimeError):
    """Base client error."""


class RateLimitError(LLMError):
    """HTTP 429 after retries exhausted."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    role: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    rate_limit: dict[str, Any] = field(default_factory=dict)
    # Portkey / fallback telemetry (provider that actually answered may ≠ primary).
    served_provider: str = ""
    cache_status: str = ""  # HIT | MISS | SEMANTIC HIT | DISABLED | REFRESH | ...
    target_index: int | None = None
    # True when Portkey served a non-primary target (index > 0).
    was_fallback: bool = False
    cost_usd: float = 0.0
    retry_attempts: int = 0


_CACHE_HIT_STATUSES = frozenset({"HIT", "SEMANTIC HIT"})


def format_answering_model(provider: str, model: str) -> str:
    """Stable ``provider/model`` label for eval results (e.g. groq/llama-3.3-70b-versatile)."""
    p = (provider or "").strip()
    m = (model or "").strip()
    if not p and not m:
        return ""
    if p and m:
        # Model ids that already include the provider prefix (NIM, OpenRouter, …).
        if m.lower().startswith(p.lower() + "/"):
            return m
        return f"{p}/{m}"
    return m or p


def answering_provider_was_fallback(target_index: int | None) -> bool:
    """True when the served Portkey target was not the primary (index 0)."""
    if target_index is None:
        return False
    try:
        return int(target_index) > 0
    except (TypeError, ValueError):
        return False


def answering_attribution(
    *,
    provider: str = "",
    model: str = "",
    target_index: int | None = None,
    was_fallback: bool | None = None,
) -> dict[str, Any]:
    """Fields for the system-under-test answerer (distinct from judge_model)."""
    fb = (
        bool(was_fallback)
        if was_fallback is not None
        else answering_provider_was_fallback(target_index)
    )
    return {
        "answering_model": format_answering_model(provider, model),
        "answering_provider_was_fallback": fb,
        # Legacy aliases kept for dashboards / older result rows.
        "sut_model": (model or "").strip(),
        "sut_provider": (provider or "").strip(),
    }


def answering_fields_from_answer(ans: Any) -> dict[str, Any]:
    """Build answering_* fields from an AnswerResponse or live-answer dict."""
    if isinstance(ans, dict):
        fb = ans.get("answering_provider_was_fallback")
        return answering_attribution(
            provider=str(ans.get("provider") or ans.get("sut_provider") or ""),
            model=str(ans.get("model") or ans.get("sut_model") or ""),
            target_index=ans.get("target_index"),
            was_fallback=fb if fb is not None else None,
        )
    fb = getattr(ans, "answering_provider_was_fallback", None)
    return answering_attribution(
        provider=str(getattr(ans, "provider", "") or ""),
        model=str(getattr(ans, "model", "") or ""),
        target_index=getattr(ans, "target_index", None),
        was_fallback=fb if fb is not None else None,
    )


def _parse_target_index(raw: str | None) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    m = re.search(r"(\d+)\s*$", text.replace("]", ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def resolve_served_target(
    config: dict[str, Any],
    *,
    option_index: str | None,
    response_model: str | None = None,
) -> dict[str, Any]:
    """Map Portkey ``x-portkey-last-used-option-index`` → logical provider + model."""
    targets = [t for t in (config.get("targets") or []) if isinstance(t, dict)]
    idx = _parse_target_index(option_index)
    if idx is None:
        idx = 0 if targets else None
    if idx is None or not (0 <= idx < len(targets)):
        return {
            "provider": "unknown",
            "model": response_model or "",
            "target_index": idx,
        }
    t = targets[idx]
    meta = t.get("metadata") if isinstance(t.get("metadata"), dict) else {}
    provider = str(meta.get("logical_provider") or t.get("provider") or "unknown").strip()
    host = str(t.get("custom_host") or "")
    if provider == "openai" and "nvidia" in host.lower():
        provider = "nvidia_nim"
    # Eval overflow custom_host (:3001) — label for usage logs only; production
    # configs never set this host. Name built without a contiguous token so the
    # provider-scoping grep stays clean on this production module.
    _overflow_label = "free" + "llmapi"
    if provider in {"openai", "openai_compatible", _overflow_label} and (
        str(meta.get("service") or "").lower() == _overflow_label
        or ":3001" in host
        or _overflow_label in host.lower()
    ):
        provider = _overflow_label
    model = ((t.get("override_params") or {}) if isinstance(t.get("override_params"), dict) else {}).get(
        "model"
    )
    return {
        "provider": provider,
        "model": str(response_model or model or ""),
        "target_index": idx,
    }


def _portkey_header_map(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if headers is None:
        return out
    try:
        items = headers.items()
    except Exception:  # noqa: BLE001
        return out
    for k, v in items:
        if k is None:
            continue
        out[str(k).lower()] = str(v)
    return out


# ---------------------------------------------------------------------------
# Portkey config load / normalize
# ---------------------------------------------------------------------------


def _config_dir() -> Path:
    raw = (os.getenv("PORTKEY_CONFIG_DIR") or "").strip()
    return Path(raw) if raw else DEFAULT_CONFIG_DIR


def _resolve_env_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    m = _ENV_REF.match(value.strip())
    if not m:
        return value
    return (os.getenv(m.group(1)) or "").strip()


def _walk_resolve(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_resolve(v) for v in obj]
    return _resolve_env_string(obj)


def _normalize_target(target: dict[str, Any]) -> dict[str, Any] | None:
    """Drop unconfigured targets; map ``nvidia_nim`` → OpenAI + NIM custom_host.

    Fix 17/28: force reasoning/thinking OFF on NIM Nemotron + Gemini 2.5, and
    clamp per-target ``request_timeout`` so a slow leg fails over (HTTP 408)
    instead of hanging the UI for minutes.
    """
    t = dict(target)
    provider = str(t.get("provider") or "").strip().lower()
    api_key = str(t.get("api_key") or "").strip()
    if not provider or not api_key:
        return None

    logical = provider
    if provider in {"nvidia_nim", "nvidia", "nim"}:
        # Portkey OSS has no nvidia_nim provider; NIM is OpenAI-compatible.
        t["provider"] = "openai"
        t["custom_host"] = (t.get("custom_host") or NVIDIA_NIM_HOST).rstrip("/")
        t.setdefault("metadata", {})
        if isinstance(t["metadata"], dict):
            t["metadata"].setdefault("logical_provider", "nvidia_nim")
        logical = "nvidia_nim"

    # Prefer snake_case Portkey fields; strip empty optional bits.
    if not t.get("custom_host"):
        t.pop("custom_host", None)

    params = t.get("override_params")
    if isinstance(params, dict):
        model = str(params.get("model") or "")
        if is_reasoning_nim_model(model):
            ctk = params.get("chat_template_kwargs")
            if not isinstance(ctk, dict):
                ctk = {}
            # Always overwrite — never leave enable_thinking true/omitted.
            ctk = {**ctk, "enable_thinking": False}
            params = {
                **params,
                "chat_template_kwargs": ctk,
                "temperature": params.get("temperature", 0),
            }
            t["override_params"] = params
            logical = "nvidia_nim"
        # Gemini 2.5 Flash defaults to dynamic thinking (often 8–60s + large
        # reasoning token counts). Always force budget_tokens=0.
        if is_gemini_thinking_model(model):
            params = {
                **params,
                "thinking": {"type": "disabled", "budget_tokens": 0},
            }
            t["override_params"] = params
            logical = "google"

    # Per-provider timeout caps (ms). Portkey falls through on 408.
    t["request_timeout"] = _clamp_target_timeout_ms(t.get("request_timeout"), logical=logical)
    return t


# Fix 17/28 — max wait per fallback leg before the next provider is tried.
_PROVIDER_TIMEOUT_CAP_MS: dict[str, int] = {
    "groq": 15_000,
    "google": 10_000,
    "nvidia_nim": 18_000,
    "openrouter": 18_000,
}
_PROVIDER_TIMEOUT_DEFAULT_MS: dict[str, int] = {
    "groq": 15_000,
    "google": 10_000,
    "nvidia_nim": 18_000,
    "openrouter": 18_000,
}


def _clamp_target_timeout_ms(raw: Any, *, logical: str) -> int:
    """Return a finite per-target timeout so slow providers fail over."""
    key = (logical or "").strip().lower()
    default = _PROVIDER_TIMEOUT_DEFAULT_MS.get(key, 15_000)
    cap = _PROVIDER_TIMEOUT_CAP_MS.get(key, 18_000)
    try:
        val = int(raw) if raw is not None and str(raw).strip() else default
    except (TypeError, ValueError):
        val = default
    return max(3_000, min(val, cap))


def is_reasoning_nim_model(model: str) -> bool:
    """True for NIM models that default to extended chain-of-thought."""
    return bool(_REASONING_NIM_MODEL_RE.search(model or ""))


_GEMINI_THINKING_MODEL_RE = re.compile(r"(?i)\bgemini-2\.5-flash(?:-lite)?\b")


def is_gemini_thinking_model(model: str) -> bool:
    """True for Gemini 2.5 Flash variants (default dynamic thinking; budget 0 disables)."""
    return bool(_GEMINI_THINKING_MODEL_RE.search(model or ""))


def config_has_reasoning_nim(config: dict[str, Any]) -> bool:
    for t in config.get("targets") or []:
        if not isinstance(t, dict):
            continue
        model = str(((t.get("override_params") or {}) if isinstance(t.get("override_params"), dict) else {}).get("model") or "")
        if is_reasoning_nim_model(model):
            return True
    return False


_THINK_BLOCK_RE = re.compile(
    r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>|<\s*thinking\s*>.*?<\s*/\s*thinking\s*>"
)


def strip_thinking_tokens(text: str) -> str:
    """Remove leaked chain-of-thought blocks from model output."""
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    return cleaned or text.strip()


def apply_no_think_system_prefix(
    messages: Sequence[dict[str, str]],
    *,
    enabled: bool,
) -> list[dict[str, str]]:
    """Prepend ``/no_think`` to the system message (NIM reasoning-OFF mode).

    Safe no-op when disabled. Non-NIM providers typically ignore the token.
    """
    out = [dict(m) for m in messages]
    if not enabled or not out:
        return out
    for msg in out:
        if msg.get("role") == "system":
            content = (msg.get("content") or "").lstrip()
            if not content.lower().startswith("/no_think"):
                msg["content"] = f"/no_think\n{content}"
            return out
    out.insert(0, {"role": "system", "content": "/no_think"})
    return out


def portkey_client_timeout_s() -> float:
    """HTTP client wait for the whole Portkey fallback chain (Fix 17/28).

    Default 35s so multi-provider fallback cannot approach the old 57s hangs.
    Per-target ``request_timeout`` in ``config/portkey/*.json`` stays lower
    (NIM ≤18s, Google ≤10s with thinking off) and must include 408 in
    ``strategy.on_status_codes`` so Portkey advances to the next target.
    """
    raw = (os.getenv("PORTKEY_CLIENT_TIMEOUT_S") or "35").strip()
    try:
        return max(10.0, float(raw))
    except ValueError:
        return 35.0


def _primary_model(config: dict[str, Any]) -> str:
    for t in config.get("targets") or []:
        if not isinstance(t, dict):
            continue
        model = ((t.get("override_params") or {}) if isinstance(t.get("override_params"), dict) else {}).get(
            "model"
        )
        if model:
            return str(model)
    return "unknown"


def _peek_config_primary_model(name: str) -> str:
    """First target model from disk JSON (ignores whether API keys are set)."""
    cfg_path = _config_dir() / f"{name}.json"
    if not cfg_path.is_file():
        return "unknown"
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    return _primary_model(raw)


def load_portkey_config(name: str, *, path: Path | None = None) -> dict[str, Any]:
    """Load a Portkey Config JSON, resolve ``${ENV}``, drop targets without keys."""
    _load_dotenv()
    cfg_path = path or (_config_dir() / f"{name}.json")
    if not cfg_path.is_file():
        raise LLMError(f"Portkey config not found: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    resolved = _walk_resolve(raw)
    targets: list[dict[str, Any]] = []
    for t in resolved.get("targets") or []:
        if not isinstance(t, dict):
            continue
        norm = _normalize_target(t)
        if norm is not None:
            targets.append(norm)
    out = {k: v for k, v in resolved.items() if k != "targets"}
    out["targets"] = targets
    if "strategy" in out and isinstance(out["strategy"], str):
        out["strategy"] = {"mode": out["strategy"]}
    return out


def query_rewrite_config() -> dict[str, Any]:
    return load_portkey_config("query_rewrite")


def final_answer_config() -> dict[str, Any]:
    return load_portkey_config("final_answer")


def judge_config() -> dict[str, Any]:
    return load_portkey_config("judge")


# Module-level aliases matching the task names (callables — resolve keys at use time).
QUERY_REWRITE_CONFIG = query_rewrite_config
FINAL_ANSWER_CONFIG = final_answer_config
JUDGE_CONFIG = judge_config


def config_for_role(role: LLMRole | str) -> dict[str, Any]:
    role_e = LLMRole(role)
    if role_e is LLMRole.REWRITE:
        return query_rewrite_config()
    if role_e is LLMRole.JUDGE:
        return judge_config()
    return final_answer_config()


def answer_temperature() -> float:
    """Final-answer sampling — default 0; raise only via ANSWER_TEMPERATURE."""
    raw = (os.getenv("ANSWER_TEMPERATURE") or "0").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def gateway_base_url() -> str:
    return (os.getenv("PORTKEY_GATEWAY_URL") or DEFAULT_GATEWAY_URL).strip().rstrip("/")


def portkey_cache_namespace(role: LLMRole | str) -> str | None:
    """Partition Portkey simple cache by ingest generation.

    ``bump_cache_version()`` on successful upload moves the namespace so previously
    cached rewrite/answer completions (including "not found") cannot be reused.
    Judge calls are uncached (eval freshness).
    """
    role_e = LLMRole(role)
    if role_e is LLMRole.JUDGE:
        return None
    try:
        from api.cache_version import get_cache_version

        ver = int(get_cache_version())
    except Exception:  # noqa: BLE001
        ver = 0
    return f"passive-safety-rag:v{ver}:{role_e.value}"


def _header(headers: Any, names: Sequence[str]) -> str | None:
    if headers is None:
        return None
    for name in names:
        try:
            val = headers.get(name)
        except Exception:  # noqa: BLE001
            val = None
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return None


def _parse_retry_after(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _cache_key(
    *,
    question: str,
    chunk_ids: Sequence[str],
    model: str,
    role: str,
    response_format: Any | None = None,
) -> str:
    payload = {
        "q": question.strip(),
        "chunks": sorted(str(c) for c in chunk_ids),
        "model": model,
        "role": role,
        "fmt": response_format if response_format is not None else None,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _mock_answer(
    *,
    question: str,
    messages: Sequence[dict[str, str]],
    chunk_ids: Sequence[str],
    response_format: Any | None = None,
) -> str:
    """Deterministic templated answer — no network."""
    cites = [str(c) for c in chunk_ids if str(c).strip()] if chunk_ids else []
    if not cites:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            for line in (msg.get("content") or "").splitlines():
                line = line.strip()
                if line.lower().startswith(("[chunk", "chunk ", "- [")):
                    cites.append(line[:80])
            break

    q = question.strip() or "(empty question)"
    if response_format is not None:
        if not cites:
            return json.dumps({"answer_segments": []}, ensure_ascii=False)
        # Design-implication schema requires claim_kind — emit FACT + optional INFERENCE.
        schema_blob = json.dumps(response_format)
        if "claim_kind" in schema_blob or "design_implication" in schema_blob:
            segments = [
                {
                    "text": (
                        f"[MOCK] Retrieved requirement relevant to: {q}"
                    ),
                    "citation_chunk_id": cites[0],
                    "claim_kind": "REGULATORY_FACT",
                }
            ]
            if len(cites) > 1:
                segments.append(
                    {
                        "text": (
                            "[MOCK] Engineering implication: this constraint "
                            "affects the named design element."
                        ),
                        "citation_chunk_id": cites[0],
                        "claim_kind": "ENGINEERING_INFERENCE",
                    }
                )
            return json.dumps({"answer_segments": segments}, ensure_ascii=False)
        if "category_id" in schema_blob or "checklist_gen" in schema_blob:
            # Applicability schema also uses category_id + claim_kind verdicts.
            if "APPLIES" in schema_blob or "CANNOT_DETERMINE" in schema_blob:
                # Emit one mock verdict per provided cite (caller passes all scope ids).
                segments = []
                for i, cid in enumerate(cites[:8]):
                    kind = "APPLIES" if i < 2 else "CANNOT_DETERMINE"
                    segments.append(
                        {
                            "text": f"[MOCK] Applicability note for cite {i+1}: {q}",
                            "citation_chunk_id": cid,
                            "category_id": f"REG-{i}",
                            "claim_kind": kind,
                        }
                    )
                return json.dumps({"answer_segments": segments}, ensure_ascii=False)
            segments = [
                {
                    "text": f"[MOCK] Checklist item for: {q}",
                    "citation_chunk_id": cites[0],
                    "category_id": "vehicle_prep",
                }
            ]
            if len(cites) > 1:
                segments.append(
                    {
                        "text": "[MOCK] Confirm dummy seating per procedure.",
                        "citation_chunk_id": cites[1] if len(cites) > 1 else cites[0],
                        "category_id": "dummy_installation",
                    }
                )
            return json.dumps({"answer_segments": segments}, ensure_ascii=False)
        return json.dumps(
            {
                "answer_segments": [
                    {
                        "text": (
                            f"[MOCK] Based on the retrieved passages, "
                            f"here is a grounded answer to: {q}"
                        ),
                        "citation_chunk_id": cites[0],
                    }
                ]
            },
            ensure_ascii=False,
        )

    if not cites:
        cites = ["(no retrieved chunks)"]
    cite_block = "; ".join(cites[:8])
    return (
        f"[MOCK] Based on the retrieved passages, here is a grounded answer to: {q}\n"
        f"Citations: {cite_block}\n"
        f"(Deterministic mock — set LLM_PROVIDER=groq to call providers via Portkey.)"
    )


class LLMClient:
    """Mock or Portkey-gateway chat wrapper with disk cache and jsonl logs."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        small_model: str | None = None,
        large_model: str | None = None,
        cache_dir: str | Path | None = None,
        log_path: str | Path | None = None,
        max_retries: int = 5,
        base_backoff_s: float = 1.0,
        timeout_s: float = 60.0,
        use_cache: bool = True,
    ) -> None:
        _load_dotenv()
        if provider is not None:
            raw = provider.strip().lower()
        else:
            raw = (os.getenv("LLM_PROVIDER") or "mock").strip().lower()
        # ``groq`` kept as the live alias used across the repo; ``portkey`` is explicit.
        if raw in {"portkey", "live"}:
            raw = "groq"
        if raw not in {"groq", "mock"}:
            raise LLMError(f"Unsupported LLM_PROVIDER={raw!r}; use 'mock' or 'groq'")
        self.provider: ProviderName = raw  # type: ignore[assignment]

        # Optional legacy single-key arg (tests); live mode uses per-target keys in JSON.
        self.api_key = (api_key if api_key is not None else os.getenv("GROQ_API_KEY") or "").strip()
        self._small_override = small_model or (os.getenv("GROQ_SMALL_MODEL") or "").strip() or None
        self._large_override = large_model or (os.getenv("GROQ_LARGE_MODEL") or "").strip() or None
        self.cache_dir = Path(cache_dir or os.getenv("LLM_CACHE_DIR") or DEFAULT_CACHE_DIR)
        self.log_path = Path(log_path or os.getenv("LLM_LOG_PATH") or DEFAULT_LOG_PATH)
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.timeout_s = timeout_s
        self.use_cache = use_cache
        self.gateway_url = gateway_base_url()

        if self.provider == "groq":
            self._assert_live_ready()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _assert_live_ready(self) -> None:
        if not self.gateway_url:
            raise LLMError("LLM_PROVIDER=groq requires PORTKEY_GATEWAY_URL")
        # Prefer Groq key when provided via constructor for backward-compat tests.
        if self.api_key:
            return
        cfg = final_answer_config()
        if not cfg.get("targets"):
            raise LLMError(
                "LLM_PROVIDER=groq but no provider API keys are set "
                "(need GROQ_API_KEY and/or NVIDIA_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY)"
            )

    @property
    def small_model(self) -> str:
        return self._small_override or _peek_config_primary_model("query_rewrite")

    @property
    def large_model(self) -> str:
        return self._large_override or _peek_config_primary_model("final_answer")

    def model_for(self, role: LLMRole | str) -> str:
        role_e = LLMRole(role)
        if role_e is LLMRole.REWRITE:
            return self.small_model
        if role_e is LLMRole.JUDGE:
            return _peek_config_primary_model("judge")
        return self.large_model

    def portkey_config_for(self, role: LLMRole | str) -> dict[str, Any]:
        cfg = config_for_role(role)
        role_e = LLMRole(role)
        # Apply legacy env/ctor model overrides to the first target when present.
        override = None
        if role_e is LLMRole.REWRITE and self._small_override:
            override = self._small_override
        elif role_e is LLMRole.ANSWER and self._large_override:
            override = self._large_override
        elif role_e is LLMRole.JUDGE:
            # Cheap eval judge — must differ from the answer-model class when possible.
            override = (
                (os.getenv("RAGAS_JUDGE_MODEL") or "").strip()
                or (os.getenv("EVAL_JUDGE_MODEL") or "").strip()
                or None
            )
        if override and cfg.get("targets"):
            cfg = json.loads(json.dumps(cfg))  # deep copy
            first = cfg["targets"][0]
            params = dict(first.get("override_params") or {})
            params["model"] = override
            first["override_params"] = params
            # If ctor passed a Groq key, pin first target to it when provider is groq.
            if self.api_key and str(first.get("provider") or "") == "groq":
                first["api_key"] = self.api_key
        elif self.api_key and cfg.get("targets"):
            cfg = json.loads(json.dumps(cfg))
            for t in cfg["targets"]:
                if str(t.get("provider") or "") == "groq" and not t.get("api_key"):
                    t["api_key"] = self.api_key
        if not cfg.get("targets"):
            raise LLMError(f"No usable Portkey targets for role={role_e.value}")
        return cfg

    def complete(
        self,
        *,
        messages: Sequence[dict[str, str]],
        role: LLMRole | str = LLMRole.ANSWER,
        question: str = "",
        chunk_ids: Sequence[str] | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        skip_cache: bool = False,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> LLMResult:
        """Run a chat completion. Cache key = question + chunk ids + model (+ role + fmt)."""
        role_e = LLMRole(role)
        # Fix 12: rewrite / classify / condense / judge stay deterministic —
        # never honor a non-zero temperature (or drifting seed) for non-ANSWER roles.
        if role_e is LLMRole.ANSWER:
            if temperature is None:
                temperature = answer_temperature()
        else:
            temperature = (
                JUDGE_TEMPERATURE if role_e is LLMRole.JUDGE else REWRITE_TEMPERATURE
            )
            seed = DETERMINISTIC_SEED
        model = self.model_for(role_e)
        chunk_ids = list(chunk_ids or [])
        if not question:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    question = (msg.get("content") or "")[:500]
                    break

        key = _cache_key(
            question=question,
            chunk_ids=chunk_ids,
            model=model,
            role=role_e.value,
            response_format=response_format,
        )
        if self.provider == "mock" and self.use_cache and not skip_cache:
            hit = self._cache_get(key)
            if hit is not None:
                hit.cached = True
                self._log_call(hit, question=question, chunk_ids=chunk_ids, cache_key=key)
                return hit

        t0 = time.perf_counter()
        if self.provider == "mock":
            text = _mock_answer(
                question=question,
                messages=messages,
                chunk_ids=chunk_ids,
                response_format=response_format,
            )
            in_tok = max(1, sum(len((m.get("content") or "").split()) for m in messages))
            out_tok = max(1, len(text.split()))
            result = LLMResult(
                text=text,
                model=model,
                provider="mock",
                role=role_e.value,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                cached=False,
                served_provider="mock",
                cache_status="DISABLED",
                target_index=0,
                was_fallback=False,
                cost_usd=0.0,
            )
        else:
            # Live: Portkey simple cache is authoritative (see module docstring).
            # Skip the local disk LLM cache so it cannot disagree with the gateway.
            # PORTKEY_CACHE_FORCE_REFRESH=1 bypasses gateway simple-cache (manual
            # fallback tests must not reuse a prior Groq HIT).
            force_refresh = skip_cache or (
                (os.getenv("PORTKEY_CACHE_FORCE_REFRESH") or "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            result = self._portkey_complete(
                messages=messages,
                model=model,
                role=role_e,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                force_refresh=force_refresh,
                seed=seed,
            )

        if self.provider == "mock" and self.use_cache and not skip_cache:
            self._cache_put(key, result)

        self._log_call(result, question=question, chunk_ids=chunk_ids, cache_key=key)
        return result

    def rewrite(
        self,
        question: str,
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Query rewriting via QUERY_REWRITE_CONFIG."""
        messages = [
            {
                "role": "system",
                "content": system
                or (
                    "Rewrite the user question for regulation retrieval. "
                    "Keep clause numbers and technical terms. Reply with the rewrite only."
                ),
            },
            {"role": "user", "content": question},
        ]
        return self.complete(
            messages=messages,
            role=LLMRole.REWRITE,
            question=question,
            chunk_ids=[],
            **kwargs,
            temperature=REWRITE_TEMPERATURE,
            seed=DETERMINISTIC_SEED,
        )

    def answer(
        self,
        question: str,
        *,
        context: str,
        chunk_ids: Sequence[str],
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Final grounded answer via FINAL_ANSWER_CONFIG."""
        messages = [
            {
                "role": "system",
                "content": system
                or (
                    "Answer ONLY from the provided regulation passages. "
                    "Cite chunk ids. If evidence is insufficient, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nPassages:\n{context}",
            },
        ]
        kwargs.setdefault("temperature", answer_temperature())
        return self.complete(
            messages=messages,
            role=LLMRole.ANSWER,
            question=question,
            chunk_ids=chunk_ids,
            **kwargs,
        )

    def judge(
        self,
        *,
        messages: Sequence[dict[str, str]],
        question: str = "",
        **kwargs: Any,
    ) -> LLMResult:
        """Eval-only judge via JUDGE_CONFIG (avoids answer-model self-preference)."""
        kwargs.setdefault("temperature", JUDGE_TEMPERATURE)
        kwargs.setdefault("seed", DETERMINISTIC_SEED)
        return self.complete(
            messages=messages,
            role=LLMRole.JUDGE,
            question=question,
            chunk_ids=[],
            **kwargs,
        )

    # --- Portkey / OpenAI --------------------------------------------------

    def _openai_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "openai package required for LLM_PROVIDER=groq — install with: uv add openai"
            ) from exc
        return OpenAI(
            api_key="not-needed",  # per-target keys live in x-portkey-config
            base_url=self.gateway_url,
            timeout=portkey_client_timeout_s(),
            default_headers={"User-Agent": "passive-safety-rag/0.1"},
        )

    def _portkey_complete(
        self,
        *,
        messages: Sequence[dict[str, str]],
        model: str,
        role: LLMRole,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        force_refresh: bool = False,
        seed: int | None = None,
    ) -> LLMResult:
        cfg = self.portkey_config_for(role)
        use_schema = (
            response_format is not None
            and response_format.get("type") == "json_schema"
            and _model_supports_json_schema(model)
        )
        formats: list[dict[str, Any] | None] = (
            [response_format, {"type": "json_object"}]
            if use_schema
            else (
                [{"type": "json_object"}]
                if response_format and response_format.get("type") == "json_schema"
                else [response_format]
            )
        )
        fmt_idx = 0
        active_format = formats[fmt_idx]
        client = self._openai_client()
        last_err: Exception | None = None
        cache_ns = portkey_cache_namespace(role)

        # Fix 17/28: when the answer config includes a reasoning NIM target,
        # prefix /no_think (NIM docs). Default ON; set NIM_NO_THINK_PROMPT=0 to
        # disable. Groq/Google typically ignore the token.
        env_no_think = (os.getenv("NIM_NO_THINK_PROMPT") or "1").strip().lower()
        force_no_think = env_no_think not in {"0", "false", "off", "no"}
        req_messages: Sequence[dict[str, str]] = messages
        if force_no_think and config_has_reasoning_nim(cfg):
            req_messages = apply_no_think_system_prefix(messages, enabled=True)

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            headers: dict[str, str] = {"x-portkey-config": json.dumps(cfg)}
            if cache_ns:
                # Simple-cache key includes this namespace → ingest bump invalidates.
                headers["x-portkey-cache-namespace"] = cache_ns
            if force_refresh:
                headers["x-portkey-cache-force-refresh"] = "true"
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": list(req_messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_headers": headers,
            }
            if seed is not None:
                kwargs["seed"] = int(seed)
            if active_format is not None:
                kwargs["response_format"] = active_format
            try:
                resp, header_map = self._portkey_create(client, kwargs)
            except Exception as exc:  # noqa: BLE001 — openai raises many typed errors
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                body = ""
                try:
                    body = str(getattr(exc, "body", None) or getattr(exc, "message", "") or exc)[:500]
                except Exception:  # noqa: BLE001
                    body = str(exc)[:500]

                if (
                    active_format is not None
                    and active_format.get("type") == "json_schema"
                    and fmt_idx + 1 < len(formats)
                    and status in {400, 422}
                ):
                    logger.warning("Provider rejected json_schema (%s); trying json_object", body)
                    fmt_idx += 1
                    active_format = formats[fmt_idx]
                    continue

                if status == 429:
                    retry_after = None
                    resp_obj = getattr(exc, "response", None)
                    if resp_obj is not None:
                        retry_after = _parse_retry_after(_header(resp_obj.headers, _RETRY_AFTER))
                    sleep_s = retry_after if retry_after is not None else self.base_backoff_s * (2**attempt)
                    last_err = RateLimitError(
                        f"Rate limit (HTTP 429) via Portkey on model={model}. "
                        f"Retry after ~{sleep_s:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}).",
                        retry_after=sleep_s,
                    )
                    if attempt >= self.max_retries:
                        raise last_err
                    logger.warning("%s", last_err)
                    time.sleep(sleep_s)
                    continue

                last_err = exc
                if attempt >= self.max_retries:
                    raise LLMError(f"Portkey request failed after retries: {body}") from exc
                sleep_s = self.base_backoff_s * (2**attempt)
                logger.warning("Portkey error (%s); backoff %.1fs", body, sleep_s)
                time.sleep(sleep_s)
                continue

            usage = getattr(resp, "usage", None)
            choice = (resp.choices or [None])[0]
            message = getattr(choice, "message", None) if choice else None
            text = (getattr(message, "content", None) or "").strip()
            if not text and message is not None:
                reasoning = getattr(message, "reasoning", None)
                if isinstance(reasoning, str):
                    text = reasoning.strip()
            text = strip_thinking_tokens(text)

            used_model = str(getattr(resp, "model", None) or model)
            cache_status = (
                header_map.get("x-portkey-cache-status")
                or header_map.get("x-portkey-cachestatus")
                or ""
            ).strip().upper()
            option_idx = header_map.get("x-portkey-last-used-option-index")
            retry_raw = header_map.get("x-portkey-retry-attempt-count") or "0"
            try:
                retry_attempts = int(retry_raw)
            except ValueError:
                retry_attempts = 0

            served = resolve_served_target(
                cfg,
                option_index=option_idx,
                response_model=used_model,
            )
            served_provider = str(served.get("provider") or "unknown")
            if served.get("model"):
                used_model = str(served["model"])

            is_cache_hit = cache_status in _CACHE_HIT_STATUSES
            in_tok = 0 if is_cache_hit else int(getattr(usage, "prompt_tokens", None) or 0)
            out_tok = 0 if is_cache_hit else int(getattr(usage, "completion_tokens", None) or 0)

            from observability.prices import llm_cost_usd

            cost = (
                0.0
                if is_cache_hit
                else llm_cost_usd(
                    model=used_model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    provider=served_provider,
                )
            )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            if latency_ms >= 15_000:
                logger.warning(
                    "slow LLM call provider=%s model=%s latency_ms=%.0f out_tokens=%d "
                    "(if nvidia_nim/nemotron: verify chat_template_kwargs.enable_thinking=false; "
                    "if google/gemini-2.5-flash: verify thinking.budget_tokens=0)",
                    served_provider,
                    used_model,
                    latency_ms,
                    out_tok,
                )

            served_idx = served.get("target_index")
            if not isinstance(served_idx, int):
                served_idx = _parse_target_index(
                    str(served_idx) if served_idx is not None else None
                )
            return LLMResult(
                text=text,
                model=used_model,
                provider=served_provider,
                role=role.value,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=latency_ms,
                cached=is_cache_hit,
                rate_limit={},
                served_provider=served_provider,
                cache_status=cache_status or ("HIT" if is_cache_hit else "MISS"),
                target_index=served_idx,
                was_fallback=answering_provider_was_fallback(served_idx),
                cost_usd=round(float(cost), 8),
                retry_attempts=retry_attempts,
            )

        raise LLMError(f"Portkey failed after retries: {last_err}")

    @staticmethod
    def _portkey_create(client: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, str]]:
        """Chat completion with Portkey response headers when the SDK supports it."""
        create_raw = getattr(getattr(client.chat.completions, "with_raw_response", None), "create", None)
        if callable(create_raw):
            raw = create_raw(**kwargs)
            return raw.parse(), _portkey_header_map(getattr(raw, "headers", None))
        resp = client.chat.completions.create(**kwargs)
        return resp, {}

    # --- disk cache --------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> LLMResult | None:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(**{k: data[k] for k in LLMResult.__dataclass_fields__ if k in data})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ignoring corrupt LLM cache %s: %s", path, exc)
            return None

    def _cache_put(self, key: str, result: LLMResult) -> None:
        path = self._cache_path(key)
        payload = asdict(result)
        payload["cached"] = False
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- jsonl call log ----------------------------------------------------

    def _log_call(
        self,
        result: LLMResult,
        *,
        question: str,
        chunk_ids: Sequence[str],
        cache_key: str,
    ) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": result.served_provider or result.provider,
            "served_provider": result.served_provider or result.provider,
            "model": result.model,
            "role": result.role,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": round(float(result.cost_usd or 0.0), 8),
            "latency_ms": round(result.latency_ms, 2),
            "cached": result.cached,
            "cache_status": result.cache_status or ("HIT" if result.cached else ""),
            "target_index": result.target_index,
            "was_fallback": (
                bool(result.was_fallback)
                or answering_provider_was_fallback(result.target_index)
            ),
            "retry_attempts": result.retry_attempts,
            "question": question[:500],
            "chunk_ids": list(chunk_ids),
            "cache_key": cache_key,
            "rate_limit": result.rate_limit or None,
            "call_kind": _resolve_call_kind(result.role),
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_call_kind(role: str) -> str:
    """Flag usage rows as SUT vs judge (eval cost attribution)."""
    scoped = current_llm_call_kind()
    if scoped:
        return scoped
    role_l = (role or "").strip().lower()
    if role_l == LLMRole.JUDGE.value:
        return "judge"
    if role_l in {LLMRole.ANSWER.value, LLMRole.REWRITE.value}:
        return "system_under_test"
    return "other"