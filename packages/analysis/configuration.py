"""compare_configuration() — PRD.md PR-006, TRD.md Phase 9.

Deterministic structural diff between two run configuration dicts (seat,
restraint, airbag, dummy, contacts, solver). A pure diff cannot know
engineer *intent*, so every detected change is classified UNKNOWN — PR-006's
INTENTIONAL/DEPENDENCY/UNINTENTIONAL classifications require an engineering
changelog or reviewer input this function does not have. Guessing a
classification would violate PRD.md §8 ("must not silently ... invent").
"""

from __future__ import annotations

from typing import Any

from packages.analysis.models import ConfigDiffEntry

ALGORITHM_VERSION = "compare_configuration v0.1.0"


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = value


def compare_configuration(config_a: dict[str, Any], config_b: dict[str, Any]) -> list[ConfigDiffEntry]:
    flat_a: dict[str, Any] = {}
    flat_b: dict[str, Any] = {}
    _flatten("", config_a, flat_a)
    _flatten("", config_b, flat_b)

    paths = sorted(set(flat_a) | set(flat_b))
    entries: list[ConfigDiffEntry] = []
    for path in paths:
        a_value = flat_a.get(path)
        b_value = flat_b.get(path)
        if path not in flat_a or path not in flat_b:
            entries.append(
                ConfigDiffEntry(
                    path=path,
                    run_a_value=a_value,
                    run_b_value=b_value,
                    change_status="UNKNOWN",
                    change_classification="UNKNOWN",
                )
            )
        elif a_value == b_value:
            entries.append(
                ConfigDiffEntry(
                    path=path,
                    run_a_value=a_value,
                    run_b_value=b_value,
                    change_status="SAME",
                    change_classification="UNKNOWN",
                )
            )
        else:
            entries.append(
                ConfigDiffEntry(
                    path=path,
                    run_a_value=a_value,
                    run_b_value=b_value,
                    change_status="CHANGED",
                    change_classification="UNKNOWN",
                )
            )
    return entries


def changed_paths(diffs: list[ConfigDiffEntry]) -> list[str]:
    return [d.path for d in diffs if d.change_status in ("CHANGED", "UNKNOWN")]
