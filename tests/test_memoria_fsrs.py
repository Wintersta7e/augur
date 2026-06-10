"""Pure FSRS-exponential math for Memoria — no Redis."""

from tabula.config import AugurConfig
from memoria.fsrs import make_memory_id, normalize_severity, retrievability, review

CFG = AugurConfig()


def _state(**kw):
    base = {
        "memory_id": "x",
        "pattern": {
            "kind": "episodic",
            "domains": ["chess", "typing"],
            "rule_key": "LOW+LOW",
            "severity": "MEDIUM",
        },
        "S": 1.0,
        "D": 5.0,
        "last_review_session": 1,
        "tier": "warm",
        "status": "active",
        "origin_severity": "MEDIUM",
        "memory_kind": "episodic",
        "source_sessions": ["s1"],
    }
    base.update(kw)
    return base


def test_normalize_severity():
    assert normalize_severity("high") == "HIGH"
    assert normalize_severity(" Medium ") == "MEDIUM"
    assert normalize_severity(None) == "LOW"
    assert normalize_severity("bogus") == "LOW"


def test_make_memory_id_stable_under_domain_order_and_timestamps():
    a = make_memory_id(
        {
            "kind": "episodic",
            "domains": ["typing", "chess"],
            "rule_key": "LOW+LOW",
            "severity": "MEDIUM",
        }
    )
    b = make_memory_id(
        {
            "kind": "episodic",
            "domains": ["chess", "typing"],
            "rule_key": "LOW+LOW",
            "severity": "MEDIUM",
        }
    )
    assert a == b and len(a) == 64


def test_make_memory_id_distinct_on_rulekey_and_severity():
    base = {
        "kind": "episodic",
        "domains": ["chess", "typing"],
        "rule_key": "LOW+LOW",
        "severity": "MEDIUM",
    }
    assert make_memory_id(base) != make_memory_id({**base, "rule_key": "LOW+MEDIUM"})
    assert make_memory_id(base) != make_memory_id({**base, "severity": "HIGH"})
    assert make_memory_id(base) != make_memory_id({**base, "rule_key": None})


def test_retrievability_curve():
    assert (
        abs(retrievability(_state(S=1.0, last_review_session=1), 6, CFG) - 0.9**5)
        < 1e-9
    )
    assert retrievability(_state(last_review_session=5), 5, CFG) == 1.0
    # S_MIN guard: S=0 does not divide-by-zero
    assert 0.0 < retrievability(_state(S=0.0, last_review_session=1), 2, CFG) <= 1.0


def test_review_grows_s_and_stamps_and_caps():
    s = review(
        _state(S=1.0, source_sessions=["s1"]),
        active_session=4,
        session_id="s2",
        cfg=CFG,
    )
    assert s["S"] == 1.5
    assert s["last_review_session"] == 4
    assert s["source_sessions"] == ["s1", "s2"]
    big = review(_state(S=CFG.memory_s_max), active_session=9, session_id="s9", cfg=CFG)
    assert big["S"] == CFG.memory_s_max


def test_review_idempotent_per_session():
    st = _state(S=1.0, source_sessions=["s1", "s2"])
    out = review(st, active_session=7, session_id="s2", cfg=CFG)
    assert out is st
