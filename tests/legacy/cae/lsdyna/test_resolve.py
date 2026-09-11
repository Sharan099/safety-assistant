"""Recursive include resolution — PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md §8."""

import zipfile
from pathlib import Path

from packages.cae.lsdyna.resolve import resolve_recursive
from packages.ingestion.archives import inspect_archive


def _make_zip(path: Path, entries: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return path


def test_resolves_three_level_chain_beyond_direct_includes(tmp_path: Path) -> None:
    archive = _make_zip(
        tmp_path / "deck.zip",
        {
            "main.key": "*KEYWORD\n*INCLUDE\nlevel2.k\n*END\n",
            "level2.k": "*KEYWORD\n*INCLUDE\nlevel3.k\n*PART\nmid\n5,5,5\n*END\n",
            "level3.k": "*KEYWORD\n*PART\nleaf\n9,9,9\n*END\n",
        },
    )
    manifest = inspect_archive(archive)

    result = resolve_recursive(archive, "main.key", manifest.members)

    assert set(result.decks) == {"main.key", "level2.k", "level3.k"}
    assert result.graph.deck_status["main.key"] == "COMPLETE"
    assert result.graph.deck_status["level2.k"] == "COMPLETE"
    assert not result.truncated
    # The transitively-reached leaf's own PART is genuinely parsed, not
    # just detected as an include target.
    assert result.decks["level3.k"].parts[0].part_id == 9


def test_missing_transitive_include_is_reported_not_silently_dropped(tmp_path: Path) -> None:
    archive = _make_zip(
        tmp_path / "deck.zip",
        {
            "main.key": "*KEYWORD\n*INCLUDE\nlevel2.k\n*END\n",
            "level2.k": "*KEYWORD\n*INCLUDE\nghost.k\n*END\n",
        },
    )
    manifest = inspect_archive(archive)

    result = resolve_recursive(archive, "main.key", manifest.members)

    assert set(result.decks) == {"main.key", "level2.k"}  # ghost.k never existed to parse
    missing_edges = [e for e in result.graph.edges if e.status == "MISSING"]
    assert len(missing_edges) == 1
    assert missing_edges[0].target_as_written == "ghost.k"


def test_cycle_across_three_levels_is_detected(tmp_path: Path) -> None:
    archive = _make_zip(
        tmp_path / "deck.zip",
        {
            "a.k": "*KEYWORD\n*INCLUDE\nb.k\n*END\n",
            "b.k": "*KEYWORD\n*INCLUDE\nc.k\n*END\n",
            "c.k": "*KEYWORD\n*INCLUDE\na.k\n*END\n",
        },
    )
    manifest = inspect_archive(archive)

    result = resolve_recursive(archive, "a.k", manifest.members)

    assert set(result.decks) == {"a.k", "b.k", "c.k"}
    assert any(e.status == "CYCLE" for e in result.graph.edges)


def test_max_files_bound_truncates_a_pathological_case(tmp_path: Path) -> None:
    # A long linear chain of 5 files, capped to 3.
    entries = {f"f{i}.k": f"*KEYWORD\n*INCLUDE\nf{i + 1}.k\n*END\n" for i in range(4)}
    entries["f4.k"] = "*KEYWORD\n*END\n"
    archive = _make_zip(tmp_path / "deck.zip", entries)
    manifest = inspect_archive(archive)

    result = resolve_recursive(archive, "f0.k", manifest.members, max_files=3)

    assert result.truncated is True
    assert len(result.decks) == 3


def test_oversized_transitive_include_is_skipped_not_parsed_or_missing(tmp_path: Path) -> None:
    small_main = "*KEYWORD\n*INCLUDE\nhuge.k\n*END\n"  # ~30 bytes
    huge_content = "*KEYWORD\n*PART\ntitle\n1,1,1\n*END\n" + ("$ padding\n" * 100)  # well over 100 bytes
    archive = _make_zip(tmp_path / "deck.zip", {"main.key": small_main, "huge.k": huge_content})
    manifest = inspect_archive(archive)

    result = resolve_recursive(archive, "main.key", manifest.members, max_file_size_bytes=100)

    assert "main.key" in result.decks  # small enough, parsed normally
    assert "huge.k" not in result.decks  # never opened/parsed
    assert "huge.k" in result.skipped_too_large
    skipped_edges = [e for e in result.graph.edges if e.status == "SKIPPED_TOO_LARGE"]
    assert len(skipped_edges) == 1
    assert skipped_edges[0].target_as_written == "huge.k"


def test_ambiguous_transitive_include_is_not_guessed(tmp_path: Path) -> None:
    archive = _make_zip(
        tmp_path / "deck.zip",
        {
            "main.key": "*KEYWORD\n*INCLUDE\nvehicle.k\n*END\n",
            "subA/vehicle.k": "*KEYWORD\n*END\n",
            "subB/vehicle.k": "*KEYWORD\n*END\n",
        },
    )
    manifest = inspect_archive(archive)

    result = resolve_recursive(archive, "main.key", manifest.members)

    # Both same-named candidates are parsed (so the graph can see the real
    # collision and report it), but the edge itself is never resolved to
    # either one — the ambiguity is reported honestly, not guessed away.
    assert set(result.decks) == {"main.key", "subA/vehicle.k", "subB/vehicle.k"}
    ambiguous_edges = [e for e in result.graph.edges if e.status == "AMBIGUOUS"]
    assert len(ambiguous_edges) == 1
    assert ambiguous_edges[0].resolved_target is None
