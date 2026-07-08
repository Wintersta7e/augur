"""Screen functions — pure verdicts over text/proposals."""

from conscientia import charter
from conscientia.screens import (
    CORRECTIVE_SUFFIX,
    Verdict,
    make_violation,
    screen_advice_text,
    screen_proposal,
    screen_taught_content,
)
from tabula.config import AugurConfig

CFG = AugurConfig()


def test_clean_advice_passes():
    v = screen_advice_text("Long pauses this morning were steady thought.", CFG)
    assert v == Verdict(True)


def test_forbidden_valence_blocks_case_insensitively():
    v = screen_advice_text("You should Take A Break right now.", CFG)
    assert not v.ok
    assert v.code == "forbidden_valence"
    assert v.principle == "restraint"
    assert "take a break" in (v.detail or "")


def test_output_extra_patterns_extend():
    cfg = AugurConfig(conscientia_output_extra_patterns=("you must",))
    assert not screen_advice_text("You MUST do this now.", cfg).ok
    assert screen_advice_text("You MUST do this now.", CFG).ok


def test_disabled_screens_pass_everything():
    for cfg in (
        AugurConfig(conscientia_enabled=False),
        AugurConfig(conscientia_output_screen_enabled=False),
    ):
        assert screen_advice_text("take a break", cfg).ok


def test_non_string_and_empty_are_ok():
    assert screen_advice_text("", CFG).ok
    assert screen_advice_text(None, CFG).ok  # type: ignore[arg-type]


def test_taught_content_screens_both_fields():
    assert not screen_taught_content("please take a break daily", None, CFG).ok
    assert not screen_taught_content(None, "as an ai rule", CFG).ok
    assert screen_taught_content("mornings are deep work", "deep_work", CFG).ok


def test_taught_screen_gates():
    cfg = AugurConfig(conscientia_teach_screen_enabled=False)
    assert screen_taught_content("take a break", None, cfg).ok


def _prop(kind="escalation_rule", target="LOW+LOW", klass="safe", action=None):
    return {
        "kind": kind,
        "target": target,
        "klass": klass,
        "action": action or {"target": "HIGH"},
    }


def test_proposal_clean_passes():
    assert screen_proposal(_prop(), CFG).ok


def test_proposal_protected_surface_refused():
    v = screen_proposal(
        _prop(
            kind="gate_calibration",
            target="conscientia/charter.py",
            action={"op": "floor_set", "value": 0.1},
        ),
        CFG,
    )
    assert not v.ok and v.code == "protected_surface" and v.principle == "containment"


def test_proposal_forged_klass_refused():
    # kind maps to gated regardless of the record's claimed klass
    v = screen_proposal(_prop(kind="code", klass="safe", action={"patch": "x"}), CFG)
    assert not v.ok and v.code == "not_safe_kind"


def test_proposal_prompt_text_valence_refused():
    v = screen_proposal(
        _prop(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": "Advise them to take a break."},
        ),
        CFG,
    )
    assert not v.ok and v.code == "forbidden_valence"


def test_proposal_missing_kind_is_verdict_not_exception():
    v = screen_proposal({"target": "x", "klass": "safe", "action": {}}, CFG)
    assert not v.ok and v.code == "malformed_proposal"


def test_proposal_screen_gates():
    bad = {"kind": "code", "target": "x", "klass": "safe", "action": {}}
    assert screen_proposal(bad, AugurConfig(conscientia_enabled=False)).ok
    assert screen_proposal(
        bad, AugurConfig(conscientia_proposal_screen_enabled=False)
    ).ok


def test_taught_screen_master_flag_gates():
    assert screen_taught_content(
        "take a break", None, AugurConfig(conscientia_enabled=False)
    ).ok


def test_make_violation_shape():
    rec = make_violation(
        "advice",
        "forbidden_valence",
        "matched 'take a break'",
        "restraint",
        decision_id="d1",
        domain="typing",
        regenerated=True,
        now=123.0,
    )
    assert rec["surface"] == "advice" and rec["ts"] == 123.0
    assert rec["charter_version"] == charter.CHARTER_VERSION
    assert rec["regenerated"] is True and rec["entity"] is None


def test_corrective_suffix_has_slot():
    assert "{matched}" in CORRECTIVE_SUFFIX
