"""Conscientia invariants C1-C5 (spec S7).

Adjudicated split of C1's screens-OFF assertion: for context_directive /
gate_calibration / semantic_fact (and the gated code/structural kinds) the
proposal's `target` field is only a LABEL -- each kind's real write
destination (the dialogue-directives store, limen tuning keys, memoria
records) is a fixed store chosen by the action payload, never by `target`.
C1a covers the kinds where `target` IS the write destination (escalation
rule key / prompt domain / threshold domain); C1b covers the label/gated
kinds, where C5 (kill-switch parity with pre-Conscientia `main`) requires
those applies to be allowed to proceed with screens off, so C1b asserts the
store-level floor instead: no apply kind ever writes into
`augur:conscientia:*`.
"""

import fakeredis
import pytest

from conscientia import charter
from conscientia.review import review_gated
from imperator import apply as A, proposals as P
from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager

CFG_ON = AugurConfig()
CFG_OFF = AugurConfig(conscientia_enabled=False)

# C1a: `target` IS the write destination (matrix rule key / prompt domain /
# threshold domain) -- each has its own kind-specific validator floor
# (severity-pair key shape, no-current-prompt refusal, sigma range/presence).
_C1A_DESTINATION_KINDS = ("escalation_rule", "prompt_strategy", "sigma")

# C1b: `target` is a LABEL only; the real destination is fixed per kind and
# chosen by the action payload -- plus the gated kinds (code/structural),
# which never apply in any configuration (I1).
_C1B_LABEL_KINDS = (
    "context_directive",
    "gate_calibration",
    "semantic_fact",
    "code",
    "structural",
)


def _pm():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    return pm


class _Cfg:  # apply-path cfg (same fields as Task 7's)
    min_prompt_len = 20
    prompt_forbidden_patterns = ("take a break",)
    imperator_ii_apply_enabled = True
    imperator_ii_dedupe_staleness_s = 86400.0
    correlation_window_min_s = 5.0
    correlation_window_max_s = 120.0
    dialogue_confirmed_apply_enabled = True
    conscientia_enabled = True
    conscientia_proposal_screen_enabled = True
    conscientia_output_extra_patterns = ()
    conscientia_teach_extra_patterns = ()


def _targeting(kind: str, prefix: str) -> dict:
    """A normalized, logged proposal of `kind` whose target sits under `prefix`."""
    p = P.normalize_klass(
        P.make_proposal(
            kind=kind,
            target=f"{prefix}anything",
            action={
                "target": "MEDIUM",
                "domain": "typing",
                "text": "x" * 40,
                "patch": "x",
            },
            rationale="r",
        )
    )
    p["status"] = "logged"
    return p


@pytest.mark.parametrize("kind", sorted(P._KIND_KLASS))
@pytest.mark.parametrize("prefix", charter.PROTECTED_SURFACES)
def test_c1_screens_on_no_kind_applies_onto_protected_surfaces(kind, prefix):
    pm = _pm()
    cfg = _Cfg()
    cfg.conscientia_enabled = True
    p = _targeting(kind, prefix)
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="s")
    assert out["status"] != "applied"
    out_c = A.apply_proposal(pm, dict(p), cfg=cfg, session_id="s", confirmed=True)
    assert out_c["status"] != "applied"


# C1a: target IS the destination here, so the screens-ON floor (C1) still
# holds with screens off, via each kind's own validator (no screen involved).
@pytest.mark.parametrize("kind", _C1A_DESTINATION_KINDS)
@pytest.mark.parametrize("prefix", charter.PROTECTED_SURFACES)
def test_c1a_screens_off_destination_kinds_never_apply(kind, prefix):
    pm = _pm()
    cfg = _Cfg()
    cfg.conscientia_enabled = False
    p = _targeting(kind, prefix)
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="s")
    assert out["status"] != "applied"
    out_c = A.apply_proposal(pm, dict(p), cfg=cfg, session_id="s", confirmed=True)
    assert out_c["status"] != "applied"


# Adjudicated split of spec C1 into C1a/C1b: for these label-target kinds,
# `target` never reaches Conscientia's own writes, so C5 (kill-switch parity
# with pre-Conscientia `main`) requires the apply to proceed here -- assert
# the store-level floor instead: no apply kind writes into augur:conscientia:*.
@pytest.mark.parametrize("kind", _C1B_LABEL_KINDS)
@pytest.mark.parametrize("prefix", charter.PROTECTED_SURFACES)
def test_c1b_screens_off_label_kinds_no_conscientia_store_writes(kind, prefix):
    pm = _pm()
    cfg = _Cfg()
    cfg.conscientia_enabled = False
    p = _targeting(kind, prefix)
    A.apply_proposal(pm, dict(p), cfg=cfg, session_id="s", confirmed=True)
    assert pm._r.keys("augur:conscientia:*") == []


def test_c4_verdict_never_enables_gated_apply():
    pm = _pm()
    p = P.normalize_klass(
        P.make_proposal(
            kind="code", target="vigil/x.py", action={"patch": "x"}, rationale="r"
        )
    )
    p["status"] = "logged"
    pm.save_conscientia_verdict(review_gated(p, CFG_ON))  # verdict exists
    for confirmed in (False, True):
        out = A.apply_proposal(
            pm, dict(p), cfg=_Cfg(), session_id="s", confirmed=confirmed
        )
        assert out["status"] != "applied"


def test_c5_kill_switch_means_no_records_anywhere():
    from conscientia.screens import (
        screen_advice_text,
        screen_proposal,
        screen_taught_content,
    )

    assert screen_advice_text("take a break", CFG_OFF).ok
    assert screen_taught_content("take a break", None, CFG_OFF).ok
    assert screen_proposal(
        {"kind": "sigma", "target": "conscientia/x", "klass": "safe", "action": {}},
        CFG_OFF,
    ).ok
