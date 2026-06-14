import fakeredis
import pytest
from tabula.persistence import PersistenceManager
from imperator import apply as A, proposals as P


def _pm():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    return pm


class _Cfg:
    min_prompt_len = 20
    prompt_forbidden_patterns = ()
    imperator_ii_apply_enabled = True
    imperator_ii_dedupe_staleness_s = 86400.0


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


def test_apply_marker_failure_keeps_applied_status(monkeypatch):
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
    # The matrix write already committed; a marker failure must not flip a real
    # apply back to 'logged'/'error' (re-applying the same patch later is harmless).
    assert out["status"] == "applied"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"


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
