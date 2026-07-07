import fakeredis
from tabula.persistence import PersistenceManager
from imperator import apply as A, proposals as P


def _pm():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    return pm


class _Cfg:
    min_prompt_len = 20
    prompt_forbidden_patterns = ()
    imperator_ii_apply_enabled = False  # watch-first OFF
    dialogue_confirmed_apply_enabled = True
    imperator_ii_dedupe_staleness_s = 86400.0


def test_confirmed_applies_despite_watch_first_off():
    pm, cfg = _pm(), _Cfg()
    p = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="taught",
        source="dialogue",
    )
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="d1", confirmed=True)
    assert out["status"] == "applied"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"
    assert out["action"]["prior_target"] == "LOW"  # rollback anchor


def test_confirmed_refuses_gated():
    pm, cfg = _pm(), _Cfg()
    p = P.make_proposal(
        kind="code",
        target="x",
        action={},
        rationale="no",
        source="dialogue",
        klass="gated",
    )
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="d1", confirmed=True)
    assert out["status"] == "logged"


def test_confirmed_disabled_flag_logs_only():
    pm, cfg = _pm(), _Cfg()
    cfg.dialogue_confirmed_apply_enabled = False
    p = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="x",
        source="dialogue",
    )
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="d1", confirmed=True)
    assert out["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"


def test_confirmed_flag_never_opens_autonomous_path():
    # Reverse-direction flag independence: dialogue_confirmed_apply_enabled=True
    # must NOT substitute for imperator_ii_apply_enabled on the AUTONOMOUS path
    # (confirmed=False). Pins against a future
    # `or cfg.dialogue_confirmed_apply_enabled` creeping into the watch-first
    # gate: an otherwise fully applicable safe proposal stays 'logged' with no
    # write and no arm.
    pm, cfg = _pm(), _Cfg()
    assert cfg.dialogue_confirmed_apply_enabled is True
    assert cfg.imperator_ii_apply_enabled is False
    p = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="r",
    )
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="s1", confirmed=False)
    assert out["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"  # no write
    assert "prior_target" not in p["action"]  # no anchor
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # gate never armed


def test_confirmed_apply_arms_gate_seen_by_autonomous_dedupe():
    # Cross-path dedupe: a successful confirmed apply must write the SAME
    # applied-marker the autonomous path checks, so the autonomous cycle cannot
    # re-move the target in-window and bury the confirmed change's rollback
    # anchor. Pins 'skipped' — the exact status the autonomous dedupe path sets
    # (see test_apply_skips_when_already_applied).
    pm, cfg = _pm(), _Cfg()
    p1 = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="taught",
        source="dialogue",
    )
    out1 = A.apply_proposal(pm, p1, cfg=cfg, session_id="d1", confirmed=True)
    assert out1["status"] == "applied"
    assert pm.is_proposal_applied(p1["dedupe_key"]) is True  # marker written

    # Same (kind, target) -> same dedupe key, but a DIFFERENT action: if the
    # autonomous path did re-apply, the matrix move would be observable.
    cfg.imperator_ii_apply_enabled = True
    p2 = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "HIGH"},
        rationale="r",
    )
    assert p2["dedupe_key"] == p1["dedupe_key"]
    out2 = A.apply_proposal(pm, p2, cfg=cfg, session_id="s1", confirmed=False)
    assert out2["status"] == "skipped"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"  # unmoved


def test_confirmed_refuses_gated_klass_on_confirmable_kind():
    # Defense-in-depth on the klass guard in isolation: a confirmable KIND
    # (escalation_rule, in _CONFIRMED_APPLY_KINDS) carrying an anomalous
    # klass="gated" must be refused by the confirmed path — the klass check
    # must hold on its own, not only via the kind set.
    pm, cfg = _pm(), _Cfg()
    p = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="r",
        source="dialogue",
        klass="gated",
    )
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="d1", confirmed=True)
    assert out["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"  # no write
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed


def test_confirmed_matrix_write_failure_logs_and_leaves_matrix_unchanged(monkeypatch):
    """F11 regression: a matrix CAS-contention failure on the confirmed path
    ends the proposal "logged" (truthful) with the matrix UNCHANGED and NO
    rollback anchor -- pins the reversibility/truthfulness surface against a
    primary-write failure the confirmed path previously had no test for."""
    pm, cfg = _pm(), _Cfg()  # _pm seeds LOW+LOW -> LOW
    monkeypatch.setattr(
        A.matrix_ops, "apply_matrix_update", lambda *a, **k: {"error": "contention"}
    )
    p = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="taught",
        source="dialogue",
    )
    out = A.apply_proposal(pm, p, cfg=cfg, session_id="d1", confirmed=True)
    assert out["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"  # unchanged
    assert "prior_target" not in out["action"]  # no anchor -> undo can't fake it


def test_confirmed_apply_logs_swallowed_exception(monkeypatch, caplog):
    """F8 regression: a handler exception on the confirmed path fails to
    "logged" (truthful) AND is now logged, so operators can tell an infra fault
    from an ordinary validation rejection."""
    import logging

    pm, cfg = _pm(), _Cfg()

    def _boom(*a, **k):
        raise RuntimeError("redis down")

    monkeypatch.setattr(A, "_dispatch_confirmed", _boom)
    p = P.make_proposal(
        kind="gate_calibration",
        target="k1",
        action={"op": "self_tolerance_add", "state_key": "k1"},
        rationale="r",
        source="dialogue",
    )
    with caplog.at_level(logging.WARNING):
        out = A.apply_proposal(pm, p, cfg=cfg, session_id="d1", confirmed=True)
    assert out["status"] == "logged"
    assert any("confirmed apply failed" in r.message for r in caplog.records)
