"""R16 ELR topic filter, mega-chunk excerpt, vehicle continuity."""

from __future__ import annotations

from api.conversations import (
    VehicleContext,
    clear_conversation,
    inject_vehicle_context,
    merge_vehicle_context,
    new_conversation_id,
    update_vehicle_context_from_question,
)
from ingestion.chunk import iter_inline_clause_segments
from retrieval.context_budget import excerpt_for_query_terms
from retrieval.enumerative import (
    infer_regulation_from_topic,
    resolve_hard_regulation_filter,
)


def test_elr_topic_hard_filters_to_r16():
    q = "What are the Emergency Locking Retractor requirements?"
    assert infer_regulation_from_topic(q) == "UN-ECE-R16"
    assert resolve_hard_regulation_filter(q) == "UN-ECE-R16"


def test_retractor_topic_not_overridden_by_comparative():
    assert resolve_hard_regulation_filter("compare R94 and R95") is None
    assert (
        resolve_hard_regulation_filter("Emergency Locking Retractor under UN R16")
        == "UN-ECE-R16"
    )


def test_inline_clause_split_separates_elr_from_buckle():
    mega = (
        "6.2.2. Buckle\n\n"
        "The buckle shall be designed correctly.\n\n"
        "6.2.5. Retractors\n\n"
        "6.2.5.3. Emergency locking retractors\n\n"
        "An emergency locking retractor, when tested, shall lock at 0.45 g."
    )
    segs = iter_inline_clause_segments(mega)
    nums = [s[0] for s in segs]
    assert "6.2.2" in nums
    assert "6.2.5" in nums or "6.2.5.3" in nums
    assert any("emergency locking" in s[2].lower() for s in segs)


def test_excerpt_keeps_buried_elr_window():
    mega = (
        "6.2.2 Buckle " + ("padding " * 400) + "\n"
        "6.2.5.3. Emergency locking retractors An emergency locking retractor "
        "shall lock when deceleration reaches 0.45 g.\n" + ("tail " * 200)
    )
    out = excerpt_for_query_terms(
        mega, "Emergency Locking Retractor 0.45 g", max_tokens=120
    )
    low = out.lower()
    assert "emergency locking" in low or "0.45" in low
    # Should not be dominated by leading padding.
    assert low.count("padding") < 30


def test_vehicle_context_persists_across_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATIONS_DB", str(tmp_path / "conv.sqlite3"))
    cid = new_conversation_id()
    ctx1 = update_vehicle_context_from_question(
        cid, "Which regulations apply to the new BMW X3 EV?"
    )
    assert "M1" in ctx1.categories
    assert ctx1.powertrain == "ev"
    assert any("BMW" in m or "X3" in m for m in ctx1.model_hints + [ctx1.platform or ""])

    # Follow-up omits vehicle — context must still inject.
    ctx2 = update_vehicle_context_from_question(cid, "What about side impact?")
    assert ctx2.powertrain == "ev"
    injected = inject_vehicle_context("What about side impact?", ctx2)
    assert "vehicle:" in injected.lower() or "ev" in injected.lower() or "BMW" in injected
    clear_conversation(cid)


def test_merge_vehicle_context_overlays():
    base = VehicleContext(categories=["M1"], powertrain="ev", platform="BMW X3")
    inc = VehicleContext(mass_kg=170.0)
    merged = merge_vehicle_context(base, inc)
    assert merged.categories == ["M1"]
    assert merged.powertrain == "ev"
    assert merged.mass_kg == 170.0
