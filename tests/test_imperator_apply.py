import fakeredis
import pytest
from tabula.persistence import PersistenceManager
from tabula.provenance import LearnContext
from imperator import apply as A, proposals as P


def _pm():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    # Seeding the default matrix is a system write (as at real startup), so it
    # persists whatever the provenance mode.
    pm.save_escalation_matrix(
        {"version": "v1", "rules": {"LOW+LOW": "LOW"}}, ctx=LearnContext.system()
    )
    return pm


class _Cfg:
    min_prompt_len = 20
    prompt_forbidden_patterns = ()
    imperator_ii_apply_enabled = True
    imperator_ii_dedupe_staleness_s = 86400.0
    # read by the pre-arm escalation patch validation (window range check)
    correlation_window_min_s = 5.0
    correlation_window_max_s = 120.0
    # read by conscientia's pre-apply screen (charter.output_patterns) for
    # every prompt_strategy proposal: _Cfg has no conscientia_enabled, and
    # the screen's own gate check defaults MISSING flags to True (on), so
    # the pattern tuple is always built here -- must not AttributeError.
    conscientia_output_extra_patterns = ()


def _safe(p):
    P.normalize_klass(p)
    p["status"] = "logged"
    return p


def test_apply_disabled_logs_only():
    pm = _pm()
    cfg = _Cfg()
    cfg.imperator_ii_apply_enabled = False
    p = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )
    assert A.apply_proposal(pm, p, cfg=cfg, session_id="s1")["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"


def test_apply_escalation_rule_patches_and_marks():
    pm = _pm()
    p = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s1")
    assert out["status"] == "applied" and out["applied_session"] == "s1"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"
    # rollback anchor recorded from the committed CAS snapshot (not a separate read)
    assert p["action"]["prior_target"] == "LOW"
    assert pm.is_proposal_applied(p["dedupe_key"]) is True


def test_apply_prompt_requires_existing_current_prompt():
    pm = _pm()
    txt = "Consider developing minor pieces before the queen, calmly."
    p = _safe(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": txt},
            rationale="r",
        )
    )
    assert (
        A.apply_proposal(pm, dict(p), cfg=_Cfg(), session_id="s1")["status"] == "logged"
    )  # no current prompt -> no anchor
    pm.save_prompt("typing", "Some existing prompt that is long enough to pass.")
    assert (
        A.apply_proposal(pm, dict(p), cfg=_Cfg(), session_id="s1")["status"]
        == "applied"
    )
    assert pm.load_prompt("typing") == txt


def test_apply_prompt_records_prior_text_anchor():
    # A successful prompt apply must record the rollback anchor: the pre-apply
    # stored prompt, under action["prior_text"] — the key rollback reads to restore.
    pm = _pm()
    anchor = "Some existing prompt that is long enough to pass."
    pm.save_prompt("typing", anchor)
    txt = "Consider developing minor pieces before the queen, calmly."
    p = _safe(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": txt},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s1")
    assert out["status"] == "applied"
    assert p["action"]["prior_text"] == anchor
    assert pm.load_prompt("typing") == txt


def test_apply_prompt_loads_current_prompt_exactly_once(monkeypatch):
    # The precondition check and the rollback-anchor read must share ONE
    # load_prompt call: a second read is an extra round-trip and a TOCTOU window
    # where the anchor could be taken from a value that was never validated.
    pm = _pm()
    pm.save_prompt("typing", "Some existing prompt that is long enough to pass.")
    calls = []
    real_load = pm.load_prompt
    monkeypatch.setattr(
        pm, "load_prompt", lambda *a, **k: (calls.append(a), real_load(*a, **k))[1]
    )
    txt = "Consider developing minor pieces before the queen, calmly."
    p = _safe(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": txt},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s1")
    assert out["status"] == "applied"
    assert len(calls) == 1


def test_apply_prompt_unacceptable_text_fails_before_arming():
    # Unacceptable text (below min_prompt_len) must fail validation BEFORE the
    # gate arms: status stays 'logged', prompt untouched, no anchor recorded, and
    # the applied-marker is never set — so a corrected proposal for the same
    # target can still apply within the staleness window.
    pm = _pm()
    anchor = "Some existing prompt that is long enough to pass."
    pm.save_prompt("typing", anchor)
    p = _safe(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": "too short"},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s1")
    assert out["status"] == "logged"
    assert pm.load_prompt("typing") == anchor
    assert "prior_text" not in p["action"]
    assert pm.is_proposal_applied(p["dedupe_key"]) is False


def test_apply_sigma_is_propose_only():
    pm = _pm()
    p = P.normalize_klass(
        P.make_proposal(
            kind="sigma",
            target="typing",
            action={"sigma": 3.0},
            rationale="r",
        )
    )
    p["status"] = "logged"
    assert A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s1")["status"] == "logged"


def test_apply_skips_when_already_applied():
    pm = _pm()
    p = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )
    # Pre-marked applied -> a direct re-apply must be a no-op. Idempotency is
    # enforced inside apply_proposal itself, not only via run_cycle's pre-filter.
    pm.mark_proposal_applied(p["dedupe_key"], ttl_s=86400)
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s2")
    assert out["status"] == "skipped"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"  # unchanged


def test_apply_marker_failure_does_not_apply(monkeypatch):
    # The applied-marker is the ONLY thing arming the per-(kind,target) anti-thrash
    # gate. If it can't be written, apply must fail CLOSED: the primary write must
    # NOT commit and the status must NOT be 'applied' (so the gate is never left
    # open after a committed change). Marking precedes the matrix write.
    pm = _pm()
    p = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )

    def _boom(*a, **k):
        raise RuntimeError("redis down")

    monkeypatch.setattr(pm, "mark_proposal_applied", _boom)
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s1")
    assert out["status"] != "applied"
    # Fail-closed: no committed matrix change, no rollback anchor recorded.
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"
    assert "prior_target" not in p["action"]
    assert pm.is_proposal_applied(p["dedupe_key"]) is False


def test_apply_prompt_marker_failure_does_not_save(monkeypatch):
    # If the marker can't be written, the prompt save must NOT run, so the prior
    # prompt is never archived (rollback anchor preserved) and the gate stays
    # closed. A clean retry next window applies the change properly.
    pm = _pm()
    txt = "Consider developing minor pieces before the queen, calmly."
    anchor = "An existing prompt long enough to anchor a rollback."
    pm.save_prompt("typing", anchor)
    p = _safe(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": txt},
            rationale="r",
        )
    )

    def _boom(*a, **k):
        raise RuntimeError("redis down")

    saves = []
    monkeypatch.setattr(pm, "mark_proposal_applied", _boom)
    # Spy on save_prompt while still delegating to the real implementation.
    real_save = pm.save_prompt
    monkeypatch.setattr(
        pm,
        "save_prompt",
        lambda *a, **k: (saves.append(a), real_save(*a, **k))[1],
    )
    out = A.apply_proposal(pm, dict(p), cfg=_Cfg(), session_id="s1")
    assert out["status"] != "applied"
    assert saves == []  # save_prompt never ran -> rollback history intact
    assert pm.load_prompt("typing") == anchor  # prompt unchanged
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # marker never set

    # Marker subsystem recovers: the SAME proposal now applies cleanly (idempotent
    # path was never half-committed, so no anchor corruption).
    monkeypatch.setattr(pm, "mark_proposal_applied", lambda *a, **k: None)
    out2 = A.apply_proposal(pm, dict(p), cfg=_Cfg(), session_id="s2")
    assert out2["status"] == "applied"
    assert pm.load_prompt("typing") == txt
    assert (
        pm.get_prompt_history("typing")[0]["prompt"] == anchor
    )  # anchor archived once


def test_marker_failure_blocks_different_text_same_target(monkeypatch):
    # CRITICAL: a marker-write failure must NOT leave the gate open for a DIFFERENT
    # action on the same target within the staleness window. The bug was: the marker
    # was written AFTER a committed prompt save and its failure was swallowed, so a
    # different-text proposal for the same (kind,target) re-applied in-window and
    # buried the rollback anchor under a now-meaningless intermediate version.
    pm = _pm()
    anchor = "An existing prompt long enough to anchor a rollback safely."
    pm.save_prompt("typing", anchor)

    def _boom(*a, **k):
        raise RuntimeError("redis down")

    monkeypatch.setattr(pm, "mark_proposal_applied", _boom)

    text_a = "Develop the minor pieces before committing the queen, patiently."
    text_b = "Castle early and keep the king safe before launching any attack."
    prop_a = _safe(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": text_a},
            rationale="r",
        )
    )
    prop_b = _safe(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": text_b},
            rationale="r",
        )
    )
    # Same target -> same anti-thrash identity, even though the action text differs.
    assert prop_a["dedupe_key"] == prop_b["dedupe_key"]

    # Proposal A's marker write fails -> A must not apply.
    out_a = A.apply_proposal(pm, prop_a, cfg=_Cfg(), session_id="s1")
    assert out_a["status"] != "applied"

    # Proposal B (different text, same target) must NOT apply within the window:
    # with the gate unarmed, the only safe outcome is to apply nothing, so the
    # original anchor is never buried.
    out_b = A.apply_proposal(pm, prop_b, cfg=_Cfg(), session_id="s1")
    assert out_b["status"] != "applied"
    assert pm.load_prompt("typing") == anchor  # never moved off the anchor
    assert pm.get_prompt_history("typing") == []  # nothing archived -> anchor intact


def test_apply_escalation_window_branch_anchors_prior_from_cas():
    # The 'window'-apply branch patches rule_windows and anchors rollback to the
    # prior window value read from the committed CAS snapshot (not a separate read).
    pm = _pm()
    pm.save_escalation_matrix(
        {
            "version": "v1",
            "rules": {"LOW+LOW": "LOW"},
            "rule_windows": {"LOW+LOW": 30.0},
        }
    )
    p = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"window": 60.0},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s1")
    assert out["status"] == "applied"
    assert pm.load_escalation_matrix()["rule_windows"]["LOW+LOW"] == 60.0
    assert p["action"]["prior_window"] == 30.0  # from the committed CAS snapshot
    assert pm.is_proposal_applied(p["dedupe_key"]) is True


def test_apply_invalid_matrix_patch_failsafe():
    # An invalid matrix patch (severity not in LOW/MEDIUM/HIGH) must fail safe:
    # refused by the pre-arm validation -> status stays 'logged', matrix
    # unchanged, and (validate -> arm -> write) the anti-thrash marker is NOT
    # armed — a validation failure commits nothing, so it must not consume
    # the (kind, target) dedupe slot.
    pm = _pm()
    p = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "BANANA"},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s1")
    assert out["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"  # unchanged
    assert "prior_target" not in p["action"]  # no anchor on a failed write
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # gate NOT armed


def test_corrected_retry_applies_after_invalid_patch():
    # The corrected proposal for the SAME (kind, target) must apply in-window:
    # the malformed predecessor never armed the gate.
    pm = _pm()
    bad = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "BANANA"},
            rationale="r",
        )
    )
    assert A.apply_proposal(pm, bad, cfg=_Cfg(), session_id="s1")["status"] == "logged"
    good = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, good, cfg=_Cfg(), session_id="s1")
    assert out["status"] == "applied"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"
    assert good["action"]["prior_target"] == "LOW"
    assert pm.is_proposal_applied(good["dedupe_key"]) is True


def test_invalid_window_patch_leaves_gate_unarmed():
    # Window variant: an out-of-range window is refused before arming, and the
    # corrected window for the same target still applies in-window.
    pm = _pm()
    bad = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"window": 9999.0},  # outside [5, 120]
            rationale="r",
        )
    )
    assert A.apply_proposal(pm, bad, cfg=_Cfg(), session_id="s1")["status"] == "logged"
    assert pm.is_proposal_applied(bad["dedupe_key"]) is False
    good = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"window": 60.0},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, good, cfg=_Cfg(), session_id="s1")
    assert out["status"] == "applied"
    assert pm.load_escalation_matrix()["rule_windows"]["LOW+LOW"] == 60.0


def test_confirmed_invalid_escalation_patch_leaves_gate_unarmed():
    # The human-confirmed path shares the validate -> arm -> write ordering:
    # a malformed confirmed patch is refused (logged) without arming.
    pm = _pm()
    cfg = _Cfg()
    cfg.dialogue_confirmed_apply_enabled = True
    p = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "BANANA"},
            rationale="r",
        )
    )
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="s1", confirmed=True)
    assert out["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False


def test_apply_unknown_kind_failsafe(monkeypatch):
    # Defense-in-depth: if a kind ever passes the auto-applicable check but is not
    # one the dispatch handles, the trailing else must fail safe to 'logged' and
    # never touch the matrix. Forcing is_auto_applicable models a future kind added
    # to _AUTO_APPLY_KINDS without a matching apply branch.
    pm = _pm()
    monkeypatch.setattr(P, "is_auto_applicable", lambda p: True)
    p = P.normalize_klass(
        P.make_proposal(
            kind="future_unhandled_kind",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )
    p["status"] = "logged"
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s1")
    assert out["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"  # unchanged
    assert pm.is_proposal_applied(p["dedupe_key"]) is False


@pytest.mark.parametrize("kind", sorted(P._KIND_KLASS) + ["unknown_kind"])
def test_invariant_apply_disabled_never_applies(kind):
    # Watch-first: with apply disabled, NO kind ever reaches 'applied' and the
    # matrix is never written — for every kind.
    pm = _pm()
    cfg = _Cfg()
    cfg.imperator_ii_apply_enabled = False
    p = P.normalize_klass(
        P.make_proposal(
            kind=kind,
            target="LOW+LOW",
            action={"target": "MEDIUM", "domain": "typing", "text": "x" * 40},
            rationale="r",
        )
    )
    p["status"] = "logged"
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="s")
    assert out["status"] != "applied"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"


def test_apply_never_raises_on_unexpected_write_error(monkeypatch):
    # The outer fail-safe: an unexpected exception from the committing write must
    # be swallowed into 'logged' (apply must never raise), and no matrix change
    # may land. The gate is armed BEFORE the committing write, so after a failed
    # write the anti-thrash marker is left SET: the autonomous path SKIPS
    # re-proposing this (kind, target) until the staleness window expires -- it is
    # NOT reconsidered next cycle. (A human re-teach still applies: the confirmed
    # path deliberately bypasses the is_proposal_applied pre-check.)
    pm = _pm()
    p = _safe(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )

    def _boom(*a, **k):
        raise RuntimeError("redis down mid-write")

    monkeypatch.setattr(A.matrix_ops, "apply_matrix_update", _boom)
    out = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s1")
    assert out["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"  # unchanged
    assert "prior_target" not in p["action"]
    # Gate armed-before-write -> marker left set after the failed write; pins the
    # documented anti-thrash behavior so a reorder to write-then-arm is caught.
    assert pm.is_proposal_applied(p["dedupe_key"])


# ── provenance: the apply path must still work when enforcement is on ─────────


@pytest.fixture
def _enforcing():
    from tabula.provenance import (
        ProvenanceMode,
        get_provenance_mode,
        set_provenance_mode,
    )

    prev = get_provenance_mode()
    set_provenance_mode(ProvenanceMode.ENFORCE)
    yield
    set_provenance_mode(prev)


def _learnable(pm, sid: str) -> str:
    pm.save_session_meta(sid, origin="real", created_by="test")
    return sid


@pytest.mark.parametrize(
    "kind,target,action",
    [
        ("escalation_rule", "LOW+LOW", {"target": "MEDIUM"}),
        (
            "prompt_strategy",
            "chess",
            {"text": "Be concise and specific about the board position."},
        ),
    ],
)
def test_apply_still_commits_under_enforce(_enforcing, kind, target, action):
    """A learnable session's apply must reach its committing write.

    Every _apply_* helper arms the anti-thrash gate through _arm_gate, which
    writes the applied-marker. If the helper does not FORWARD its context, the
    marker write raises under ENFORCE, _arm_gate fails closed, and the whole
    armed-apply path dies silently behind one log line.
    """
    pm = _pm()
    sid = _learnable(pm, "s-real")
    pm.save_prompt(
        "chess", "An existing prompt long enough to pass.", ctx=LearnContext.system()
    )
    p = _safe(P.make_proposal(kind=kind, target=target, action=action, rationale="r"))
    assert A.apply_proposal(pm, p, cfg=_Cfg(), session_id=sid)["status"] == "applied"
    assert pm.is_proposal_applied(p["dedupe_key"])


def test_apply_is_withheld_under_enforce_for_a_synthetic_session():
    from tabula.provenance import (
        ProvenanceMode,
        get_provenance_mode,
        set_provenance_mode,
    )

    prev = get_provenance_mode()
    set_provenance_mode(ProvenanceMode.ENFORCE)
    try:
        pm = _pm()
        pm.save_session_meta("s-synth", origin="synthetic", created_by="test")
        p = _safe(
            P.make_proposal(
                kind="escalation_rule",
                target="LOW+LOW",
                action={"target": "MEDIUM"},
                rationale="r",
            )
        )
        A.apply_proposal(pm, p, cfg=_Cfg(), session_id="s-synth")
        # The matrix is untouched: a synthetic session cannot change real policy.
        assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"
    finally:
        set_provenance_mode(prev)
