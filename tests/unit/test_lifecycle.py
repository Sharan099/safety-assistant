import pytest

from safety_assistant.domain.regulations import (
    PIPELINE_ORDER,
    IllegalTransition,
    VersionStatus,
    is_retrievable,
    transition,
)


def test_pipeline_order_is_walkable() -> None:
    cur = PIPELINE_ORDER[0]
    for nxt in PIPELINE_ORDER[1:]:
        cur = transition(cur, nxt)
    assert cur is VersionStatus.ACTIVE


@pytest.mark.parametrize(
    "bad", [("DISCOVERED", "ACTIVE"), ("ACTIVE", "DOWNLOADED"), ("PARSED", "INDEXED"), ("SUPERSEDED", "QUARANTINED")]
)
def test_skipping_or_reversing_is_illegal(bad: tuple[str, str]) -> None:
    with pytest.raises(IllegalTransition):
        transition(*bad)


def test_every_pipeline_stage_may_fail_or_quarantine() -> None:
    for s in PIPELINE_ORDER[:-1]:
        assert transition(s, VersionStatus.FAILED) is VersionStatus.FAILED
        assert transition(s, VersionStatus.QUARANTINED) is VersionStatus.QUARANTINED


def test_forced_reprocess_only_permits_reset_to_discovered() -> None:
    assert transition("ACTIVE", "DISCOVERED", force=True) is VersionStatus.DISCOVERED
    with pytest.raises(IllegalTransition):
        transition("ACTIVE", "INDEXED", force=True)


def test_retrievability_current_vs_historical() -> None:
    assert is_retrievable("ACTIVE") and not is_retrievable("SUPERSEDED")
    assert is_retrievable("SUPERSEDED", include_superseded=True)
    for s in ("VERIFIED", "QUARANTINED", "FAILED", "INDEXED"):
        assert not is_retrievable(s, include_superseded=True)
