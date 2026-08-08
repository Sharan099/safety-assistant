"""Boundary: FreeLLMAPI / eval overflow must never become a production gateway path.

Fails loudly if ``localhost:3001`` or ``FREELLMAPI`` leak into generation/,
retrieval/, or app/ — or into Portkey configs that are not explicitly named for
eval/judge scoring. That regression would recreate the "two gateways for one
concern" problem this overflow path was designed to avoid.

Also pins ``generation/llm_client.py``'s provider list to the pre-FreeLLMAPI set
so a future ``LLM_PROVIDER=freellmapi`` (or similar) cannot sneak in unnoticed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Markers that mean "route via FreeLLMAPI / eval overflow".
# Built in pieces so this file's own source does not create false positives when
# the repo-wide scan includes tests/.
_NEEDLE_LOCAL = "localhost" + ":3001"
_NEEDLE_ENV = "FREE" + "LLMAPI"

# Production packages — zero tolerance.
_PRODUCTION_DIRS = ("generation", "retrieval", "app")

# Portkey JSON that must stay production-clean (no FreeLLMAPI target).
_PRODUCTION_PORTKEY_CONFIGS = frozenset(
    {
        "final_answer.json",
        "query_rewrite.json",
        # Shared judge.json is still the default LLMClient.judge path; FreeLLMAPI
        # lives only in eval_judge_overflow.json (legacy allowlist). Active eval
        # scoring uses eval_judge_pinned.json (single model, no FreeLLMAPI).
        "judge.json",
        "eval_judge_pinned.json",
    }
)

# config/portkey/ filenames allowed to mention FreeLLMAPI / :3001.
_EVAL_JUDGE_PORTKEY_NAME_RE = re.compile(
    r"(?i)(^|[_\-])(eval|judge).*(overflow|scoring|security)|"
    r"(overflow|scoring).*(eval|judge)|"
    r"^eval_judge_overflow\.json$"
)

# Pre-FreeLLMAPI provider surface on LLMClient (do not extend without an explicit
# production design review — eval overflow must stay out of this list).
_EXPECTED_PROVIDER_NAME_LITERAL = ("mock", "groq", "portkey")
_EXPECTED_RUNTIME_ACCEPT = frozenset({"mock", "groq"})
_EXPECTED_LIVE_ALIASES = frozenset({"portkey", "live"})

_SCAN_SUFFIXES = {".py", ".json", ".yml", ".yaml", ".toml", ".md", ".txt", ".env"}
_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "data",
    "docker",
    "frontend",
    "archive",
}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _iter_text_files(*relative_roots: str) -> list[Path]:
    out: list[Path] = []
    for rel_root in relative_roots:
        root = ROOT / rel_root
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _SCAN_SUFFIXES:
                continue
            skip = False
            for parent in path.parents:
                if parent == ROOT:
                    break
                if parent.name in _SKIP_DIR_NAMES:
                    skip = True
                    break
            if skip:
                continue
            # Timestamped eval outputs
            if "eval/results/" in _rel(path):
                continue
            out.append(path)
    return out


def _hits_in_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
    found: list[str] = []
    if _NEEDLE_LOCAL in text:
        found.append(_NEEDLE_LOCAL)
    if _NEEDLE_ENV in text:
        found.append(_NEEDLE_ENV)
    return found


def _is_allowed_eval_judge_portkey(path: Path) -> bool:
    if path.parent != ROOT / "config" / "portkey":
        return False
    return bool(_EVAL_JUDGE_PORTKEY_NAME_RE.search(path.name))


def _is_allowed_location(path: Path) -> bool:
    """Application-level homes for FreeLLMAPI markers (not production routing)."""
    rel = _rel(path)
    if rel.startswith("eval/"):
        return True
    if _is_allowed_eval_judge_portkey(path):
        return True
    # This boundary test (and sibling unit tests) must be able to name the needles.
    if rel.startswith("tests/"):
        return True
    return False


def test_production_packages_never_reference_freellmapi_or_localhost_3001():
    """generation / retrieval / app must not know about FreeLLMAPI."""
    leaks: list[str] = []
    for path in _iter_text_files(*_PRODUCTION_DIRS):
        hits = _hits_in_file(path)
        if hits:
            leaks.append(f"{_rel(path)}: {', '.join(hits)}")
    assert not leaks, (
        "FreeLLMAPI / localhost:3001 leaked into production packages "
        "(eval-only overflow must not become a second production gateway):\n  - "
        + "\n  - ".join(leaks)
    )


def test_production_portkey_configs_never_reference_freellmapi():
    """FINAL_ANSWER / QUERY_REWRITE / judge.json stay free of FreeLLMAPI targets."""
    portkey_dir = ROOT / "config" / "portkey"
    leaks: list[str] = []
    for name in sorted(_PRODUCTION_PORTKEY_CONFIGS):
        path = portkey_dir / name
        assert path.is_file(), f"missing Portkey config: {name}"
        hits = _hits_in_file(path)
        if hits:
            leaks.append(f"config/portkey/{name}: {', '.join(hits)}")
    assert not leaks, (
        "Production Portkey configs must not reference FreeLLMAPI / localhost:3001:\n  - "
        + "\n  - ".join(leaks)
    )


def test_freellmapi_markers_only_in_eval_and_named_eval_judge_portkey_configs():
    """Repo grep: needles only in eval/ + eval/judge-scoring Portkey JSON (+ tests/)."""
    # Scan the trees that could accidentally wire production traffic.
    scanned = _iter_text_files(
        "generation",
        "retrieval",
        "app",
        "api",
        "core",
        "config",
        "eval",
        "tests",
    )
    offenders: list[str] = []
    for path in scanned:
        hits = _hits_in_file(path)
        if not hits:
            continue
        if _is_allowed_location(path):
            continue
        offenders.append(f"{_rel(path)}: {', '.join(hits)}")

    assert not offenders, (
        "localhost:3001 / FREELLMAPI must appear ONLY under eval/ or "
        "config/portkey/* files named for eval/judge scoring "
        "(found outside that boundary — risk of production routing via overflow):\n  - "
        + "\n  - ".join(offenders)
    )


def test_eval_judge_overflow_config_is_the_named_allowlist_file():
    """The overflow JSON exists and is the sole Portkey home for FreeLLMAPI."""
    overflow = ROOT / "config" / "portkey" / "eval_judge_overflow.json"
    assert overflow.is_file()
    assert _is_allowed_eval_judge_portkey(overflow)
    hits = _hits_in_file(overflow)
    assert _NEEDLE_ENV in hits, "overflow config should reference FREELLMAPI_UNIFIED_KEY"
    # Host URL may be localhost:3001 and/or host.docker.internal — require the env key.
    for path in (ROOT / "config" / "portkey").glob("*.json"):
        if path.name == overflow.name:
            continue
        bad = _hits_in_file(path)
        assert not bad, (
            f"config/portkey/{path.name} is not an eval/judge-scoring overflow file "
            f"but contains {bad}"
        )


def test_llm_client_provider_list_unchanged_pre_freellmapi():
    """ProviderName / accepted LLM_PROVIDER set must not grow a FreeLLMAPI entry."""
    src = (ROOT / "generation" / "llm_client.py").read_text(encoding="utf-8")

    # Literal alias used by type checkers / docs.
    m = re.search(
        r"ProviderName\s*=\s*Literal\[([^\]]+)\]",
        src,
    )
    assert m, "ProviderName Literal[...] not found in generation/llm_client.py"
    literal_members = tuple(
        s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()
    )
    assert literal_members == _EXPECTED_PROVIDER_NAME_LITERAL, (
        f"ProviderName changed from {_EXPECTED_PROVIDER_NAME_LITERAL!r} to "
        f"{literal_members!r} — do not add freellmapi/overflow as a production provider"
    )

    # Runtime accept set after portkey/live → groq aliasing.
    m2 = re.search(
        r'if raw not in \{([^}]+)\}:\s*\n\s*raise LLMError\(f"Unsupported LLM_PROVIDER',
        src,
    )
    assert m2, "LLM_PROVIDER accept-set check not found in generation/llm_client.py"
    accept = {
        s.strip().strip("'\"") for s in m2.group(1).split(",") if s.strip()
    }
    assert accept == _EXPECTED_RUNTIME_ACCEPT, (
        f"LLM_PROVIDER accept set changed from {_EXPECTED_RUNTIME_ACCEPT!r} to "
        f"{accept!r} — production must stay mock|groq (Portkey gateway), not FreeLLMAPI"
    )

    # Live aliases that collapse to groq.
    m3 = re.search(
        r'if raw in \{([^}]+)\}:\s*\n\s*raw = "groq"',
        src,
    )
    assert m3, "portkey/live → groq alias block not found"
    aliases = {
        s.strip().strip("'\"") for s in m3.group(1).split(",") if s.strip()
    }
    assert aliases == _EXPECTED_LIVE_ALIASES, (
        f"Live provider aliases changed from {_EXPECTED_LIVE_ALIASES!r} to {aliases!r}"
    )

    # Belt-and-suspenders: llm_client itself must not mention the overflow markers.
    for needle in (_NEEDLE_LOCAL, _NEEDLE_ENV, "eval_judge_overflow", "freellmapi"):
        assert needle not in src, (
            f"generation/llm_client.py contains {needle!r} — production client must "
            "stay unaware of the eval-only FreeLLMAPI overflow path"
        )


def test_llm_client_default_configs_are_still_production_three():
    """query_rewrite / final_answer / judge loaders — no overflow alias on the client."""
    src = (ROOT / "generation" / "llm_client.py").read_text(encoding="utf-8")
    for required in (
        'load_portkey_config("query_rewrite")',
        'load_portkey_config("final_answer")',
        'load_portkey_config("judge")',
    ):
        assert required in src, f"missing production config loader call: {required}"
    assert 'load_portkey_config("eval_judge_overflow")' not in src
    assert "EVAL_JUDGE_OVERFLOW" not in src
