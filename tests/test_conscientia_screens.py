"""Screen functions — pure verdicts over text/proposals."""

import pytest

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


def test_control_bytes_are_blocked():
    # The anticipatory lane is the first non-LLM text path to vox, so a
    # spoofed publisher can smuggle raw control/ANSI bytes past the substring
    # valence screen. The structural check rejects C0/C1 controls (here an ANSI
    # clear-screen escape) before any valence matching.
    v = screen_advice_text("Rate elevated.\x1b[2J", CFG)
    assert not v.ok
    assert v.code == "control_chars"
    assert v.principle == "restraint"


def test_control_bytes_osc_sequence_blocked():
    v = screen_advice_text("hi\x1b]0;evil\x07", CFG)
    assert not v.ok
    assert v.code == "control_chars"


def test_newline_and_tab_still_pass():
    # \t and \n are legitimate in LLM advice text — they must NOT be rejected.
    assert screen_advice_text("Line one.\nLine two.\tIndented.", CFG).ok


def test_control_bytes_pass_when_screen_disabled():
    # C5 parity: with the screen off, EVERYTHING passes — even control bytes.
    for cfg in (
        AugurConfig(conscientia_enabled=False),
        AugurConfig(conscientia_output_screen_enabled=False),
    ):
        assert screen_advice_text("Rate elevated.\x1b[2J", cfg).ok


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


def test_bidi_controls_are_blocked():
    # Unicode bidirectional controls (RLO, isolates, RLM) are
    # unrenderable/deceptive codepoints in the SAME structural class as C0/C1 --
    # they reorder rendered terminal text (Trojan-Source style) past the
    # substring valence screen. Same Verdict code "control_chars".
    for ch in ("‮", "⁦", "‏"):  # RLO, LRI, RLM
        v = screen_advice_text(f"Rate elevated.{ch}", CFG)
        assert not v.ok, ch
        assert v.code == "control_chars", ch
        assert v.principle == "restraint", ch


def test_bidi_controls_pass_when_screen_disabled():
    # C5 parity: with the screen off, EVERYTHING passes -- even BiDi controls.
    for cfg in (
        AugurConfig(conscientia_enabled=False),
        AugurConfig(conscientia_output_screen_enabled=False),
    ):
        assert screen_advice_text("Rate elevated.‮", cfg).ok


# -- CONTROL_CHARS_RE boundary sweep -----------------------------------------


@pytest.mark.parametrize("ch", ["\x08", "\x0b", "\x1f", "\x7f", "\x9f", "‮"])
def test_control_char_boundary_rejected(ch):
    # Pins the exact reject edges of every excluded range in CONTROL_CHARS_RE
    # (\x00-\x08, \x0b-\x1f, \x7f-\x9f) plus one BiDi control (RLO) -- a future
    # regex edit that nudges a range boundary or drops the BiDi class fails
    # this test instead of silently widening what reaches the terminal.
    v = screen_advice_text(f"Rate elevated.{ch}", CFG)
    assert not v.ok, ch
    assert v.code == "control_chars", ch


@pytest.mark.parametrize("ch", ["\x09", "\x0a", "\x20", "\x7e", "\xa0"])
def test_control_char_boundary_allowed(ch):
    # Pins the exact allow edges: tab/newline (deliberately excluded from the
    # \x0b-\x1f range for multi-line advice text), the printable-ASCII
    # boundary bytes (space, tilde), and the first byte past the C1 block
    # (\xa0, non-breaking space -- NOT in \x7f-\x9f).
    v = screen_advice_text(f"Rate elevated.{ch}", CFG)
    assert v.ok, ch
