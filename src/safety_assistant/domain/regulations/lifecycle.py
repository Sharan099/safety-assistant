"""Version lifecycle — deterministic state machine (ENGINEERING.md §6).

Only ``ACTIVE`` versions are retrievable. Transitions are validated in code,
never left to the LLM or to callers writing arbitrary strings.
"""

from __future__ import annotations

from enum import StrEnum


class VersionStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    DOWNLOADED = "DOWNLOADED"
    VALIDATED = "VALIDATED"
    PARSED = "PARSED"
    NORMALIZED = "NORMALIZED"
    CHUNKED = "CHUNKED"
    INDEXED = "INDEXED"
    VERIFIED = "VERIFIED"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    QUARANTINED = "QUARANTINED"
    FAILED = "FAILED"


PIPELINE_ORDER: tuple[VersionStatus, ...] = (
    VersionStatus.DISCOVERED,
    VersionStatus.DOWNLOADED,
    VersionStatus.VALIDATED,
    VersionStatus.PARSED,
    VersionStatus.NORMALIZED,
    VersionStatus.CHUNKED,
    VersionStatus.INDEXED,
    VersionStatus.VERIFIED,
    VersionStatus.ACTIVE,
)

_ALLOWED: dict[VersionStatus, frozenset[VersionStatus]] = {
    **{
        s: frozenset({PIPELINE_ORDER[i + 1], VersionStatus.QUARANTINED, VersionStatus.FAILED})
        for i, s in enumerate(PIPELINE_ORDER[:-1])
    },
    VersionStatus.ACTIVE: frozenset({VersionStatus.SUPERSEDED, VersionStatus.QUARANTINED}),
    VersionStatus.SUPERSEDED: frozenset({VersionStatus.ACTIVE}),  # rollback of a bad activation
    VersionStatus.QUARANTINED: frozenset({VersionStatus.DISCOVERED}),  # explicit retry from scratch
    VersionStatus.FAILED: frozenset({VersionStatus.DISCOVERED}),
}

# Current queries see ACTIVE only; historical (as-of) queries may also see SUPERSEDED
# versions — those are still verified, indexed texts with a closed validity window.
RETRIEVABLE_CURRENT: frozenset[VersionStatus] = frozenset({VersionStatus.ACTIVE})
RETRIEVABLE_HISTORICAL: frozenset[VersionStatus] = frozenset({VersionStatus.ACTIVE, VersionStatus.SUPERSEDED})


class IllegalTransition(ValueError):
    pass


def transition(current: str | VersionStatus, target: str | VersionStatus, *, force: bool = False) -> VersionStatus:
    """`force=True` permits only the explicit reprocess reset (any state → DISCOVERED)."""
    cur, tgt = VersionStatus(current), VersionStatus(target)
    if force and tgt is VersionStatus.DISCOVERED:
        return tgt
    if tgt not in _ALLOWED[cur]:
        raise IllegalTransition(f"{cur} -> {tgt} is not allowed")
    return tgt


def is_retrievable(status: str | VersionStatus, *, include_superseded: bool = False) -> bool:
    allowed = RETRIEVABLE_HISTORICAL if include_superseded else RETRIEVABLE_CURRENT
    return VersionStatus(status) in allowed
