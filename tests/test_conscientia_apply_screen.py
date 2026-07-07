"""S4 — conscientia screen in front of both apply paths (fail-closed)."""

import fakeredis
from unittest.mock import patch

from imperator import apply as A, proposals as P
from tabula.persistence import PersistenceManager


def _pm():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    return pm


class _Cfg:
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


def _safe(p):
    P.normalize_klass(p)
    p["status"] = "logged"
    return p


def _prompt_prop(text):
    return _safe(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": text},
            rationale="r",
        )
    )


def test_screened_prompt_text_refused_before_arm():
    pm = _pm()
    pm.save_prompt("typing", "existing prompt text that is long enough")
    p = _prompt_prop("Please take a break whenever this fires.")
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s")
    assert out["status"] == "logged"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed
    viols = pm.load_conscientia_violations(limit=5)
    assert viols and viols[0]["surface"] == "apply"


def test_clean_apply_still_works():
    pm = _pm()
    p = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s")
    assert out["status"] == "applied"


def test_screen_disabled_restores_old_behavior():
    pm = _pm()
    pm.save_prompt("typing", "existing prompt text that is long enough")
    cfg = _Cfg()
    cfg.conscientia_enabled = False
    p = _prompt_prop("Please take a break whenever this fires.")
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="s")
    # prompt-safety inside _apply_prompt_strategy still refuses the text —
    # the pre-existing floor — but no conscientia violation is recorded
    assert out["status"] == "logged"
    assert pm.load_conscientia_violations(limit=5) == []


def test_screen_exception_fails_closed():
    pm = _pm()
    p = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )
    with patch.object(A.screens, "screen_proposal", side_effect=RuntimeError("boom")):
        out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s")
    assert out["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"


def test_confirmed_path_screened_too():
    # A protected-surface target is refused ONLY by the conscientia screen —
    # gate_calibration's own handler would apply this floor_set otherwise,
    # so this test fails if the screen is unwired from _apply_confirmed.
    pm = _pm()
    p = _safe(
        P.make_proposal(
            kind="gate_calibration",
            target="conscientia/charter.py",
            action={"op": "floor_set", "state_key": "s1", "value": 0.1},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s", confirmed=True)
    assert out["status"] == "logged"
    viols = pm.load_conscientia_violations(limit=5)
    assert viols and viols[0]["surface"] == "apply"
    assert viols[0]["code"] == "protected_surface"
