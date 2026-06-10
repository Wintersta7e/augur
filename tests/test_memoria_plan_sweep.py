"""Tier classification + the deterministic sweep planner — pure, no Redis."""

from tabula.config import AugurConfig
from memoria.fsrs import make_memory_id
from memoria.tiers import classify, is_floor_protected, plan_sweep

CFG = AugurConfig()


def _state(**kw):
    base = {
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
    pat = kw.get(
        "pattern",
        {
            "kind": "episodic",
            "domains": ["chess", "typing"],
            "rule_key": "LOW+LOW",
            "severity": "MEDIUM",
        },
    )
    base["pattern"] = pat
    base["memory_id"] = kw.get("memory_id", make_memory_id(pat))
    return base


def _pattern(domains, rule_key="LOW+LOW", severity="MEDIUM"):
    pat = {
        "kind": "episodic",
        "domains": domains,
        "rule_key": rule_key,
        "severity": severity,
    }
    return {**pat, "memory_id": make_memory_id(pat)}


def _fake_cfg(**over):
    # plan_sweep/classify only read scalar knobs; AugurConfig bounds forbid the
    # tiny caps these tests need, so use a SimpleNamespace.
    import types

    base = dict(
        memory_prune_r=0.05,
        memory_promote_s=14,
        memory_s_growth_factor=0.5,
        memory_s_min=0.1,
        memory_s_max=365,
        max_memory_items=5000,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_classify_promote_demote_prune_keep():
    assert (
        classify(_state(S=20, tier="warm", last_review_session=10), 10, CFG)
        == "promote"
    )
    assert (
        classify(_state(S=1, tier="cold", last_review_session=1), 60, CFG) == "demote"
    )
    assert (
        classify(
            _state(S=1, tier="warm", last_review_session=1, origin_severity="MEDIUM"),
            60,
            CFG,
        )
        == "prune"
    )
    assert (
        classify(
            _state(S=1, tier="warm", last_review_session=1, origin_severity="HIGH"),
            60,
            CFG,
        )
        == "keep"
    )
    assert is_floor_protected(_state(origin_severity="HIGH")) is True


def test_plan_sweep_creates_unseen():
    plan = plan_sweep(
        [], [_pattern(["chess", "typing"])], active_session=1, session_id="s1", cfg=CFG
    )
    assert len(plan.creates) == 1
    c = plan.creates[0]
    assert c["S"] == 1.0 and c["tier"] == "warm" and c["source_sessions"] == ["s1"]
    assert c["origin_severity"] == "MEDIUM"


def test_plan_sweep_reviews_recurrence():
    existing = _state(
        S=1.0,
        source_sessions=["s1"],
        pattern={
            "kind": "episodic",
            "domains": ["chess", "typing"],
            "rule_key": "LOW+LOW",
            "severity": "MEDIUM",
        },
        last_review_session=1,
    )
    plan = plan_sweep(
        [existing],
        [_pattern(["chess", "typing"])],
        active_session=2,
        session_id="s2",
        cfg=CFG,
    )
    assert plan.creates == []
    assert len(plan.reviews) == 1 and plan.reviews[0]["S"] == 1.5
    assert plan.reviewed_count == 1


def test_plan_sweep_prunes_decayed_unprotected():
    old = _state(S=1.0, last_review_session=1, origin_severity="MEDIUM")
    plan = plan_sweep([old], [], active_session=60, session_id="s60", cfg=CFG)
    assert len(plan.prunes) == 1


def test_plan_sweep_cap_archives_lowest_r():
    cfg = _fake_cfg(max_memory_items=2)
    states = [
        _state(
            memory_id="a",
            S=50,
            last_review_session=10,
            tier="cold",
            pattern={
                "kind": "episodic",
                "domains": ["a"],
                "rule_key": None,
                "severity": "MEDIUM",
            },
        ),
        _state(
            memory_id="b",
            S=50,
            last_review_session=9,
            tier="cold",
            pattern={
                "kind": "episodic",
                "domains": ["b"],
                "rule_key": None,
                "severity": "MEDIUM",
            },
        ),
        _state(
            memory_id="c",
            S=50,
            last_review_session=1,
            tier="cold",
            pattern={
                "kind": "episodic",
                "domains": ["c"],
                "rule_key": None,
                "severity": "MEDIUM",
            },
        ),
    ]
    plan = plan_sweep(states, [], active_session=10, session_id="s10", cfg=cfg)
    pruned_ids = {s["memory_id"] for s in plan.prunes}
    assert "c" in pruned_ids and len(pruned_ids) == 1


def test_plan_sweep_cap_refuses_excess_creates():
    cfg = _fake_cfg(max_memory_items=1)
    obs = [_pattern(["a"]), _pattern(["b"])]  # 2 fresh, cap 1, no existing
    plan = plan_sweep([], obs, active_session=1, session_id="s1", cfg=cfg)
    assert len(plan.creates) == 1 and plan.refused == 1


def test_endpoint_high_floor_protected():
    st = _state(
        S=1.0,
        last_review_session=1,
        tier="warm",
        origin_severity="MEDIUM",
        pattern={
            "kind": "episodic",
            "domains": ["chess", "typing"],
            "rule_key": "HIGH+LOW",
            "severity": "MEDIUM",
        },
    )
    assert is_floor_protected(st) is True
    assert classify(st, 60, CFG) == "keep"
