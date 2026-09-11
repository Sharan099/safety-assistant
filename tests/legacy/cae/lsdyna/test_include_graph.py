import pathlib

from packages.cae.lsdyna.include_graph import build_include_graph
from packages.cae.lsdyna.models import ParsedDeck
from packages.cae.lsdyna.parser import parse_deck

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _parse_all(*names: str) -> dict[str, ParsedDeck]:
    decks: dict[str, ParsedDeck] = {}
    for name in names:
        text = FIXTURES.joinpath(name).read_text(encoding="utf-8")
        decks[name] = parse_deck(text, name)
    return decks


def test_two_level_nesting_resolves_cleanly() -> None:
    decks = _parse_all("nested_include.key", "level2.k", "level3.k")
    graph = build_include_graph(decks)

    by_parent = {(e.parent, e.target_as_written): e for e in graph.edges}
    top_edge = by_parent[("nested_include.key", "level2.k")]
    assert top_edge.status == "RESOLVED"
    assert top_edge.resolved_target == "level2.k"

    mid_edge = by_parent[("level2.k", "level3.k")]
    assert mid_edge.status == "RESOLVED"
    assert mid_edge.resolved_target == "level3.k"

    assert graph.deck_status["nested_include.key"] == "COMPLETE"
    assert graph.deck_status["level2.k"] == "COMPLETE"
    assert graph.deck_status["level3.k"] == "COMPLETE"  # no includes at all


def test_missing_include_is_reported_not_silently_dropped() -> None:
    decks = _parse_all("nested_include.key")  # level2.k deliberately not provided
    graph = build_include_graph(decks)

    edge = graph.edges[0]
    assert edge.status == "MISSING"
    assert edge.resolved_target is None
    assert graph.deck_status["nested_include.key"] == "INCOMPLETE"


def test_cycle_is_detected() -> None:
    deck_a = parse_deck("*KEYWORD\n*INCLUDE\nb.k\n*END\n", "a.k")
    deck_b = parse_deck("*KEYWORD\n*INCLUDE\na.k\n*END\n", "b.k")
    graph = build_include_graph({"a.k": deck_a, "b.k": deck_b})

    statuses = {e.status for e in graph.edges}
    assert "CYCLE" in statuses


def test_duplicate_include_of_same_target_is_flagged() -> None:
    deck = parse_deck("*KEYWORD\n*INCLUDE\nchild.k\n*INCLUDE\nchild.k\n*END\n", "parent.k")
    child = parse_deck("*KEYWORD\n*END\n", "child.k")
    graph = build_include_graph({"parent.k": deck, "child.k": child})

    statuses = [e.status for e in graph.edges]
    assert statuses == ["RESOLVED", "DUPLICATE"]


def test_absolute_path_include_is_flagged_outside_root() -> None:
    deck = parse_deck("*KEYWORD\n*INCLUDE\n/etc/passwd\n*END\n", "a.k")
    graph = build_include_graph({"a.k": deck})
    assert graph.edges[0].status == "OUTSIDE_ROOT"
    assert graph.edges[0].resolved_target is None


def test_relative_dotdot_include_resolves_normally_not_outside_root() -> None:
    # Real NHTSA decks routinely use "../../subfolder/file.k" to reach a
    # sibling folder inside the same archive — this is ordinary navigation,
    # not a traversal attempt, and must resolve like anything else since
    # resolution here is by basename match, never by following the literal
    # path on disk (see include_graph.py's _escapes_root docstring — found
    # by smoke-testing against a real 39-include NHTSA deck).
    parent = parse_deck("*KEYWORD\n*INCLUDE\n../../00_INCLUDES/01_DAB/airbag.k\n*END\n", "sub/a.k")
    target = parse_deck("*KEYWORD\n*END\n", "00_INCLUDES/01_DAB/airbag.k")
    graph = build_include_graph({"sub/a.k": parent, "00_INCLUDES/01_DAB/airbag.k": target})
    assert graph.edges[0].status == "RESOLVED"
    assert graph.edges[0].resolved_target == "00_INCLUDES/01_DAB/airbag.k"


def test_ambiguous_same_basename_in_multiple_decks_is_not_silently_guessed() -> None:
    parent = parse_deck("*KEYWORD\n*INCLUDE\nvehicle.k\n*END\n", "parent.k")
    candidate_1 = parse_deck("*KEYWORD\n*END\n", "subA/vehicle.k")
    candidate_2 = parse_deck("*KEYWORD\n*END\n", "subB/vehicle.k")
    graph = build_include_graph({"parent.k": parent, "subA/vehicle.k": candidate_1, "subB/vehicle.k": candidate_2})

    assert graph.edges[0].status == "AMBIGUOUS"
    assert graph.edges[0].resolved_target is None


def test_deck_with_no_includes_is_trivially_complete() -> None:
    decks = _parse_all("simple.k")
    graph = build_include_graph(decks)
    assert graph.deck_status["simple.k"] == "COMPLETE"
    assert graph.edges == []
