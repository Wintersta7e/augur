from imperator import proposals as P


def test_make_proposal_default_source():
    p = P.make_proposal(kind="sigma", target="chess", action={}, rationale="r")
    assert p["source"] == "imperator_ii"


def test_make_proposal_dialogue_source():
    p = P.make_proposal(
        kind="sigma",
        target="chess",
        action={},
        rationale="r",
        source="dialogue",
    )
    assert p["source"] == "dialogue"


def test_new_safe_kinds_classified():
    assert P._KIND_KLASS["context_directive"] == "safe"
    assert P._KIND_KLASS["semantic_fact"] == "safe"


def test_confirmed_apply_kinds_excludes_gated():
    assert "code" not in P._CONFIRMED_APPLY_KINDS
    assert "structural" not in P._CONFIRMED_APPLY_KINDS
    assert "observe_more" not in P._CONFIRMED_APPLY_KINDS  # safe but not actionable
    assert {
        "escalation_rule",
        "prompt_strategy",
        "sigma",
        "gate_calibration",
        "context_directive",
        "semantic_fact",
    } <= P._CONFIRMED_APPLY_KINDS
