from packages.cae.lsdyna.registry import registry_roots, root_for, structured_roots


def test_registry_roots_include_all_prd_minimum_keywords() -> None:
    roots = set(registry_roots())
    expected = {
        "NODE",
        "ELEMENT",
        "PART",
        "SECTION",
        "MAT",
        "CONTACT",
        "BOUNDARY",
        "CONSTRAIN",
        "CONTROL",
        "DATABASE",
        "DEFINE",
        "PARAMETER",
        "INCLUDE",
    }
    assert expected <= roots


def test_structured_roots_are_the_ones_with_dedicated_tables() -> None:
    assert structured_roots() == {"PART", "SECTION", "MAT", "CONTACT", "CONTROL", "DATABASE", "INCLUDE"}
    # NODE/ELEMENT are detected (in registry_roots) but not structured —
    # TRD_LEVEL3.md §19 has no cae_nodes/cae_elements table.
    assert "NODE" not in structured_roots()


def test_root_for_matches_exact_and_underscore_variants() -> None:
    assert root_for("PART") == "PART"
    assert root_for("PART_COMPOSITE") == "PART"
    assert root_for("MAT_024") == "MAT"
    assert root_for("SOMETHING_ELSE_ENTIRELY") is None
