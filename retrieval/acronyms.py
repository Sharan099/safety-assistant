"""Passive-safety acronym expansions for retrieval query rewriting.

Primary source: ``config/acronyms.json`` (Fix 5 — seeded from regulation text,
agent-reviewed; ``production_approved`` stays false until engineer sign-off).

Built-in ``DEFAULT_ACRONYMS`` is the fallback if the config is missing.
``extract_acronyms_from_text`` harvests candidates for review; it never
auto-merges into the production table.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "acronyms.json"

# Fallback if config is absent — kept in sync with verified R94/R95/R16 entries.
DEFAULT_ACRONYMS: dict[str, str] = {
    "HPC": "Head Performance Criterion",
    "HPC36": "Head Performance Criterion 36 ms",
    "HIC": "Head Injury Criterion",
    "HIC15": "Head Injury Criterion 15 ms",
    "HIC36": "Head Injury Criterion 36 ms",
    "ThCC": "Thorax Compression Criterion",
    "THCC": "Thorax Compression Criterion",
    "VC": "Viscous Criterion",
    "RDC": "Rib Deflection Criterion",
    "PSPF": "Pubic Symphysis Peak Force",
    "APF": "Abdominal Peak Force",
    "FFC": "Femur Force Criterion",
    "TCFC": "Tibia Compressive Force Criterion",
    "TI": "Tibia Index",
    "TTI": "Thoracic Trauma Index",
    "NIC": "Neck Injury Criteria",
    "MDB": "mobile deformable barrier",
    "R94": "UN Regulation No. 94 frontal collision",
    "R95": "UN Regulation No. 95 lateral collision",
    "R16": "UN Regulation No. 16 safety-belts",
    "R129": "UN Regulation No. 129 enhanced child restraint systems",
    "CRS": "child restraint system",
    "ECRS": "enhanced child restraint system",
    "CRF": "Child Restraint Fixture",
    "SBR": "seat-belt reminder",
    "ELR": "Emergency Locking Retractor",
    "ATD": "anthropomorphic test device",
    "REESS": "Rechargeable Electrical Energy Storage System",
    "H-point": "H point",
    "H-Point": "H point",
}

_DEFAULT_PHRASE_SYNONYMS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?i)\bRib\s+Deflection\b(?!\s+Criterion)"),
        "Rib Deflection Criterion (RDC)",
    ),
    (
        re.compile(r"(?i)\bPubic\s+Symphysis(?:\s+Peak)?\s+Force\b(?!\s*\()"),
        "Pubic Symphysis Peak Force (PSPF)",
    ),
    (
        re.compile(r"(?i)\bHead\s+Performance\b(?!\s+Criterion)"),
        "Head Performance Criterion (HPC)",
    ),
    (
        re.compile(r"(?i)\bAbdominal\s+Peak\s+Force\b(?!\s*\()"),
        "Abdominal Peak Force (APF)",
    ),
    # R94 normative text uses HPC; engineers often ask HIC/HIC15.
    (
        re.compile(r"(?i)\bHIC\s*15\b"),
        "HIC15 (Head Injury Criterion) HPC (Head Performance Criterion)",
    ),
    (
        re.compile(r"(?i)\bHIC\b(?!\s*\d)(?!\s*\()"),
        "HIC (Head Injury Criterion) HPC (Head Performance Criterion)",
    ),
    (
        re.compile(r"(?i)\banthropomorphic\s+test\s+device\b"),
        "anthropomorphic test device (ATD) Hybrid III dummy",
    ),
]

# Optional harvest helper for ingestion/audit (does not auto-merge into ACRONYMS).
_EXTRACT_RE = re.compile(
    r"\b((?:[Tt]he\s+)?[A-Z][A-Za-z0-9][A-Za-z0-9 /-]{2,80}?)"
    r"\s*\(([A-Za-z][A-Za-z0-9*]{0,11})\)"
)

_CACHE: tuple[float, dict[str, str], list[tuple[re.Pattern[str], str]], dict] | None = None


def acronyms_config_path() -> Path:
    raw = (os.getenv("ACRONYMS_PATH") or "").strip()
    return Path(raw) if raw else DEFAULT_CONFIG_PATH


def _compile_phrase_synonyms(rows: list[dict]) -> list[tuple[re.Pattern[str], str]]:
    out: list[tuple[re.Pattern[str], str]] = []
    for row in rows:
        match = str(row.get("match") or "").strip()
        repl = str(row.get("replacement") or "").strip()
        if not match or not repl:
            continue
        flags = str(row.get("flags") or "i").lower()
        f = re.IGNORECASE if "i" in flags else 0
        try:
            out.append((re.compile(match, f), repl))
        except re.error as exc:
            logger.warning("bad phrase_synonym pattern %r: %s", match, exc)
    return out


def load_acronym_table(
    *, path: Path | None = None, force: bool = False
) -> tuple[dict[str, str], list[tuple[re.Pattern[str], str]], dict]:
    """Load ``(acronyms, phrase_synonyms, meta)`` from config (cached by mtime)."""
    global _CACHE
    cfg = path or acronyms_config_path()
    try:
        mtime = cfg.stat().st_mtime
    except OSError:
        logger.warning("acronyms config missing: %s — using defaults", cfg)
        return dict(DEFAULT_ACRONYMS), list(_DEFAULT_PHRASE_SYNONYMS), {}

    if (
        not force
        and _CACHE is not None
        and _CACHE[0] == mtime
    ):
        return dict(_CACHE[1]), list(_CACHE[2]), dict(_CACHE[3])

    data = json.loads(cfg.read_text(encoding="utf-8"))
    meta = dict(data.get("_meta") or {})
    raw_acro = data.get("acronyms") or {}
    table: dict[str, str] = {}
    for k, v in raw_acro.items():
        key = str(k).strip()
        if isinstance(v, dict):
            exp = str(v.get("expansion") or v.get("full") or "").strip()
        else:
            exp = str(v).strip()
        if key and exp:
            table[key] = exp
    if not table:
        table = dict(DEFAULT_ACRONYMS)

    phrases = _compile_phrase_synonyms(list(data.get("phrase_synonyms") or []))
    if not phrases:
        phrases = list(_DEFAULT_PHRASE_SYNONYMS)

    if meta.get("production_approved") is False:
        logger.info(
            "acronyms lexicon loaded (agent_reviewed, production_approved=false) path=%s",
            cfg,
        )

    _CACHE = (mtime, table, phrases, meta)
    return dict(table), list(phrases), meta


def get_acronyms(*, force: bool = False) -> dict[str, str]:
    table, _phrases, _meta = load_acronym_table(force=force)
    return table


# Back-compat: module-level dict always reflects the current config (or defaults).
def _refresh_module_acronyms() -> dict[str, str]:
    return get_acronyms()


ACRONYMS: dict[str, str] = dict(DEFAULT_ACRONYMS)
try:
    ACRONYMS = get_acronyms()
except Exception:  # noqa: BLE001
    ACRONYMS = dict(DEFAULT_ACRONYMS)


def expand_phrase_synonyms(text: str) -> str:
    """Map informal metric names to canonical injury-criterion phrases."""
    if not text:
        return text
    _table, phrases, _meta = load_acronym_table()
    out = text
    for pattern, replacement in phrases:
        if replacement.lower() in out.lower():
            continue
        out = pattern.sub(replacement, out, count=1)
    return out


def expand_acronyms(
    text: str,
    *,
    acronyms: Mapping[str, str] | None = None,
) -> str:
    """Word-boundary expand known acronyms: ``VC`` → ``VC (Viscous Criterion)``.

    Longer keys win first (HIC15 before HIC). Already-expanded forms
    ``ACRONYM (Expansion)`` are left alone. Also applies phrase synonyms
    (e.g. ``Rib Deflection`` → ``Rib Deflection Criterion (RDC)``).
    """
    if not text:
        return text
    table = dict(acronyms) if acronyms is not None else get_acronyms()
    # Keep module alias in sync for tests that import ACRONYMS.
    global ACRONYMS
    if acronyms is None:
        ACRONYMS = table
    out = expand_phrase_synonyms(text)
    for key in sorted(table, key=len, reverse=True):
        expansion = table[key]
        already = re.compile(
            rf"\b{re.escape(key)}\s*\(\s*{re.escape(expansion)}\s*\)",
            re.IGNORECASE,
        )
        if already.search(out):
            continue
        pattern = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)

        def repl(m: re.Match[str], k: str = key, exp: str = expansion) -> str:
            before = out[: m.start()]
            if before.rfind("(") > before.rfind(")"):
                return m.group(0)
            return f"{m.group(0)} ({exp})"

        out = pattern.sub(repl, out)
    return out


def extract_acronyms_from_text(text: str) -> dict[str, str]:
    """Harvest ``Full Name (ACRONYM)`` pairs from regulation prose.

    Useful during ingestion/audit; does not mutate the production table.
    """
    found: dict[str, str] = {}
    if not text:
        return found
    for match in _EXTRACT_RE.finditer(text):
        full = " ".join(match.group(1).split()).strip(" .,;:")
        full = re.sub(r"^(?i:the|a|an)\s+", "", full).strip()
        acronym = match.group(2).strip().replace(" ", "")
        if re.fullmatch(r"[A-Za-z](?:\s*\*\s*[A-Za-z])+", acronym):
            acronym = re.sub(r"[^A-Za-z]", "", acronym)
        if len(acronym) < 2 or len(full) < 4:
            continue
        if acronym not in found or len(full) > len(found[acronym]):
            found[acronym] = full
    return found
