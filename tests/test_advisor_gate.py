"""Unit tests for reasoning.advisor_gate — GateDecision + build_signature.

Task 2.1: GateDecision (frozen dataclass), the fire/suppress/downgrade
constructors (id minted only when None), as_fire id-preservation, and
build_signature determinism per spec §5.
"""

from __future__ import annotations

import dataclasses

import pytest

from reasoning.advisor_gate import GateDecision, build_signature
from tests.conftest import SINGLE_MEDIUM


# ── build_signature (spec §5) ────────────────────────────────────────────────


def test_single_signature() -> None:
    p = {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": 2.0,
            "severity": "medium",
        },
    }
    s = build_signature(p)
    assert s.severity == "medium" and s.path == "single"
    assert s.state_key == "single:typing:user"
    assert not s.exempt and s.severity_score == 1.0
    assert s.ungateable is False


def test_correlation_exempt_keys_off_involved_domains() -> None:
    p = {
        "combined_severity": "HIGH",
        "correlation_found": True,
        "involved_domains": ["typing", "chess"],
        "correlated_events": [{"domain": "chess"}],
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": 3.0,
            "severity": "high",
        },
    }
    s = build_signature(p)
    assert s.exempt
    assert s.state_key == "correlation:chess+typing"  # sorted involved_domains
    assert s.severity_score == 2.0


def test_missing_entity_marks_ungateable() -> None:
    p = {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {"domain": "typing", "value": 1.0, "severity": "medium"},
    }
    s = build_signature(p)
    assert s.entity in (None, "?")
    assert s.ungateable is True


def test_question_mark_entity_marks_ungateable() -> None:
    p = {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": "typing",
            "entity": "?",
            "value": 1.0,
            "severity": "medium",
        },
    }
    s = build_signature(p)
    assert s.ungateable is True


def test_empty_entity_marks_ungateable() -> None:
    p = {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": "typing",
            "entity": "",
            "value": 1.0,
            "severity": "medium",
        },
    }
    s = build_signature(p)
    assert s.ungateable is True


def test_correlation_medium_not_exempt() -> None:
    # exempt requires correlation_found AND severity == "high"
    p = {
        "combined_severity": "MEDIUM",
        "correlation_found": True,
        "involved_domains": ["typing", "chess"],
        "correlated_events": [{"domain": "chess"}],
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": 2.5,
            "severity": "medium",
        },
    }
    s = build_signature(p)
    assert not s.exempt
    assert s.path == "correlation"
    assert s.state_key == "correlation:chess+typing"


def test_high_single_not_exempt() -> None:
    # exempt requires correlation_found; a standalone high is NOT exempt
    p = {
        "combined_severity": "HIGH",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": 4.5,
            "severity": "high",
        },
    }
    s = build_signature(p)
    assert not s.exempt
    assert s.path == "single"
    assert s.severity_score == 2.0


def test_severity_normalized_to_lowercase() -> None:
    s = build_signature(
        {
            "combined_severity": "HIGH",
            "correlation_found": False,
            "primary_anomaly": {
                "domain": "chess",
                "entity": "user",
                "value": 4.0,
                "severity": "HIGH",
            },
        }
    )
    assert s.severity == "high"


def test_signature_is_frozen() -> None:
    s = build_signature(
        {
            "combined_severity": "MEDIUM",
            "correlation_found": False,
            "primary_anomaly": {
                "domain": "chess",
                "entity": "user",
                "value": 2.0,
                "severity": "medium",
            },
        }
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.severity = "high"  # type: ignore[misc]


def test_build_signature_is_deterministic() -> None:
    p = {
        "combined_severity": "HIGH",
        "correlation_found": True,
        "involved_domains": ["chess", "typing"],
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": 3.0,
            "severity": "high",
        },
    }
    assert build_signature(p) == build_signature(p)


def test_signature_carries_escalation_rule_for_correlation() -> None:
    """The credibility class for a correlation keys off escalation_rule (spec §5)."""
    p = {
        "combined_severity": "MEDIUM",
        "correlation_found": True,
        "involved_domains": ["typing", "chess"],
        "escalation_rule": "typing+chess->high",
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": 2.5,
            "severity": "medium",
        },
    }
    s = build_signature(p)
    assert s.escalation_rule == "typing+chess->high"


def test_signature_escalation_rule_none_for_single() -> None:
    """A single event has no escalation_rule (the class falls back to domain:severity)."""
    s = build_signature(SINGLE_MEDIUM)
    assert s.escalation_rule is None


# ── GateDecision constructors (spec §4) ──────────────────────────────────────


def test_gatedecision_is_frozen() -> None:
    d = GateDecision.fire("normal")
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.action = "suppress"  # type: ignore[misc]


def test_fire_has_all_fields() -> None:
    d = GateDecision.fire("normal", deciding_arm="none", metrics={"x": 1}, tier=2)
    assert d.action == "fire"
    assert d.reason == "normal"
    assert d.deciding_arm == "none"
    assert d.metrics == {"x": 1}
    assert d.tier == 2
    assert d.probe is False
    assert d.withheld_reason is None
    assert d.mrt_eligible is False
    assert d.p_fire is None
    assert d.p_withhold is None


def test_suppress_has_action_suppress() -> None:
    d = GateDecision.suppress("habituated", deciding_arm="habituation")
    assert d.action == "suppress"
    assert d.deciding_arm == "habituation"


def test_downgrade_has_action_downgrade() -> None:
    d = GateDecision.downgrade(
        "cost_tier_downgrade_note", deciding_arm="cost_tier", tier=1
    )
    assert d.action == "downgrade"
    assert d.tier == 1


def test_id_minted_full_uuid4_hex_when_none() -> None:
    d = GateDecision.fire("normal")
    # uuid4().hex is 32 lowercase hex chars (no dashes, not truncated)
    assert len(d.id) == 32
    assert all(c in "0123456789abcdef" for c in d.id)


def test_id_unique_per_decision() -> None:
    assert GateDecision.fire("normal").id != GateDecision.fire("normal").id


def test_id_used_when_provided() -> None:
    d = GateDecision.suppress("habituated", id="fixed-id-123")
    assert d.id == "fixed-id-123"


def test_as_fire_preserves_id() -> None:
    d = GateDecision.suppress("habituated", deciding_arm="habituation")
    f = d.as_fire("gate_error_fail_open")
    assert f.action == "fire"
    assert f.reason == "gate_error_fail_open"
    assert f.id == d.id  # id PRESERVED across the conversion


def test_as_fire_resets_suppress_specific_fields() -> None:
    d = GateDecision.suppress("habituated", deciding_arm="habituation")
    f = d.as_fire("cap_fail_open")
    assert f.action == "fire"
    assert f.deciding_arm == "habituation"  # carried for audit


# ── Gate.evaluate skeleton (Task 3.1, spec §3/§4) ────────────────────────────

from tests.conftest import (  # noqa: E402
    EXEMPT_PAYLOAD,
    SINGLE_MEDIUM_NEWKEY_THAT_WOULD_SUPPRESS,
)


def test_exempt_always_fires(fake_pm, cfg) -> None:
    from reasoning.advisor_gate import Gate

    g = Gate()
    s = build_signature(EXEMPT_PAYLOAD)
    d = g.evaluate(s, fake_pm, cfg, now=100.0)
    assert d.action == "fire"
    assert d.deciding_arm == "danger_exemption"


def test_master_disabled_fires(fake_pm, cfg_disabled) -> None:
    from reasoning.advisor_gate import Gate

    g = Gate()
    s = build_signature(SINGLE_MEDIUM)
    assert g.evaluate(s, fake_pm, cfg_disabled, now=100.0).action == "fire"


def test_evaluate_is_readonly(fake_pm, cfg) -> None:
    # property: evaluate performs no Redis writes
    from reasoning.advisor_gate import Gate

    g = Gate()
    s = build_signature(SINGLE_MEDIUM)
    g.evaluate(s, fake_pm, cfg, now=100.0)
    assert fake_pm.write_calls == 0  # fake_pm counts any save_*/hset/set/lpush/sadd


def test_evaluate_is_not_a_coroutine() -> None:
    import inspect

    from reasoning.advisor_gate import Gate

    assert inspect.iscoroutinefunction(Gate.evaluate) is False  # §11 no-await proxy


def test_exempt_does_no_state_reads(fake_pm, cfg) -> None:
    from reasoning.advisor_gate import Gate

    g = Gate()
    s = build_signature(EXEMPT_PAYLOAD)
    g.evaluate(s, fake_pm, cfg, now=100.0)
    assert fake_pm.read_calls == 0  # §2(B): exempt reads no gate state


def test_no_arms_passes_all(fake_pm, cfg) -> None:
    from dataclasses import replace

    from reasoning.advisor_gate import Gate

    # A medium that trips no suppressor reaches the terminal passed_all_arms
    # fire.  The reservoir arm (Arm 5) holds a fresh single+medium until it has
    # accumulated evidence, so latch this channel committed (suppressing=False,
    # leaked count above OFF) to clear it — all other arms see unseen state.
    # Disable the Phase-2 cost_tier modifier so this stays focused on Phase-1
    # pass-through (else a one-off single+medium would be Tier-1-downgraded).
    cfg = replace(cfg, gate_cost_tier_enabled=False)
    fake_pm.save_reservoir(
        "single:chess:user", {"count": 2.0, "last_ts": 100.0, "suppressing": False}
    )
    g = Gate()  # default arm pipeline
    s = build_signature(SINGLE_MEDIUM)
    d = g.evaluate(s, fake_pm, cfg, now=100.0)
    assert d.action == "fire"
    assert d.reason == "passed_all_arms"


def test_cap_fail_open(fake_pm_at_cap, cfg) -> None:
    # A stub arm WOULD suppress a new state_key, but channel_stats is at
    # MAX_GATE_STATE_KEYS → the SUPPRESS converts to FIRE("cap_fail_open").
    from reasoning.advisor_gate import Gate

    def _always_suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("would_suppress", deciding_arm="stub")

    g = Gate(arms=[_always_suppress])
    s = build_signature(SINGLE_MEDIUM_NEWKEY_THAT_WOULD_SUPPRESS)
    d = g.evaluate(s, fake_pm_at_cap, cfg, now=100.0)
    assert d.action == "fire"
    assert d.reason == "cap_fail_open"


def test_cap_fail_open_preserves_decision_id(fake_pm_at_cap, cfg) -> None:
    from reasoning.advisor_gate import Gate

    captured: dict[str, GateDecision] = {}

    def _always_suppress(gate, sig, state, config, now, rng):
        d = GateDecision.suppress("would_suppress", deciding_arm="stub")
        captured["suppress"] = d
        return d

    g = Gate(arms=[_always_suppress])
    s = build_signature(SINGLE_MEDIUM_NEWKEY_THAT_WOULD_SUPPRESS)
    d = g.evaluate(s, fake_pm_at_cap, cfg, now=100.0)
    assert d.id == captured["suppress"].id  # id PRESERVED through cap conversion


def test_suppress_passes_through_when_trackable(fake_pm, cfg) -> None:
    # When the channel IS trackable, a suppressing arm's SUPPRESS is returned
    # as-is (no cap conversion).
    from reasoning.advisor_gate import Gate

    def _always_suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("would_suppress", deciding_arm="stub")

    g = Gate(arms=[_always_suppress])
    s = build_signature(SINGLE_MEDIUM)
    d = g.evaluate(s, fake_pm, cfg, now=100.0)
    assert d.action == "suppress"
    assert d.reason == "would_suppress"


# ── record_* + still_starved (Task 6.1, spec §4/§5/§6) ───────────────────────

from tests.conftest import (  # noqa: E402
    SINGLE_HIGH_TYPING,
)


def _gate() -> "object":
    from reasoning.advisor_gate import Gate

    return Gate()


# -- record_delivery_success: non-probe normal delivery ----------------------


def test_delivery_success_appends_gating_visible_emission(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    d = GateDecision.fire("normal", tier=2, mrt_eligible=True, p_fire=0.1)
    g.record_delivery_success(s, fake_pm, 100.0, decision=d, tier=2)

    emissions = fake_pm.load_emissions(limit=10)
    assert len(emissions) == 1
    e = emissions[0]
    assert e["state_key"] == "single:chess:user"
    assert e["decision_id"] == d.id
    assert e["probe"] is False
    assert e["audit_only"] is False
    # IPW fields persisted on the emission (spec §6 emissions schema).
    assert e["mrt_eligible"] is True
    assert e["p_fire"] == 0.1


def test_delivery_success_advances_habituation(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    d = GateDecision.fire("normal")
    g.record_delivery_success(s, fake_pm, 100.0, decision=d, tier=2)

    hab = fake_pm.load_habituation("single:chess:user")
    assert hab["h"] > 0.0  # h advanced from unseen (0) toward 1
    assert hab["last_event_ts"] == 100.0
    assert hab["count"] == 1

    # A second delivery advances h further (EWMA toward 1).
    g.record_delivery_success(
        s, fake_pm, 200.0, decision=GateDecision.fire("n"), tier=2
    )
    hab2 = fake_pm.load_habituation("single:chess:user")
    assert hab2["h"] > hab["h"]
    assert hab2["count"] == 2


def test_delivery_success_advances_advice_rate(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    g.record_delivery_success(
        s, fake_pm, 100.0, decision=GateDecision.fire("n"), tier=2
    )
    rate = fake_pm.load_advice_rate()
    assert rate != {}
    assert rate["rate_ewma"] > 0.0
    assert rate["last_ts"] == 100.0


def test_delivery_success_resets_channel_stats(fake_pm, cfg) -> None:
    # Pre-seed a starved channel; a delivery must reset the suppression streak.
    fake_pm.save_channel_stats(
        "single:chess:user",
        {"consecutive_suppressions": 5, "suppression_streak_started_ts": 50.0},
    )
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    g.record_delivery_success(
        s, fake_pm, 100.0, decision=GateDecision.fire("n"), tier=2
    )
    stats = fake_pm.load_channel_stats("single:chess:user")
    assert stats["consecutive_suppressions"] == 0
    assert stats["suppression_streak_started_ts"] is None
    assert stats["last_delivery_ts"] == 100.0


def test_delivery_success_appends_observed(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)  # value 2.0, severity medium
    g.record_delivery_success(
        s, fake_pm, 100.0, decision=GateDecision.fire("n"), tier=2
    )
    obs = fake_pm.load_observed("single:chess:user", limit=10)
    assert len(obs) == 1
    assert obs[0]["value"] == 2.0
    assert obs[0]["severity"] == "medium"


def test_delivery_success_updates_cost_tier_memory(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    g.record_delivery_success(
        s, fake_pm, 100.0, decision=GateDecision.fire("n"), tier=2
    )
    mem = fake_pm.load_cost_tier_memory("single:chess:user")
    assert mem["count"] == 1
    assert "earned_tier2" in mem

    # A tier-2 delivery marks earned_tier2 True (online cost_tier signal).
    g.record_delivery_success(
        s, fake_pm, 200.0, decision=GateDecision.fire("n"), tier=2
    )
    mem2 = fake_pm.load_cost_tier_memory("single:chess:user")
    assert mem2["count"] == 2
    assert mem2["earned_tier2"] is True


# -- record_delivery_success: probe (gating-invisible) -----------------------


def test_delivery_success_probe_appends_probe_emission_only(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    d = GateDecision.fire(
        "bet_hedge_probe",
        tier=2,
        probe=True,
        withheld_reason="habituated",
        mrt_eligible=True,
        p_fire=0.1,
        p_withhold=0.9,
    )
    g.record_delivery_success(s, fake_pm, 100.0, decision=d, tier=2)

    emissions = fake_pm.load_emissions(limit=10)
    assert len(emissions) == 1
    e = emissions[0]
    assert e["probe"] is True
    assert e["withheld_reason"] == "habituated"
    # IPW fields on the probe row so it joins its withheld sibling (spec §6/§9).
    assert e["mrt_eligible"] is True
    assert e["p_fire"] == 0.1

    # Probe must NOT advance h / rate / starvation / observed.
    assert fake_pm.load_habituation("single:chess:user") == {}
    assert fake_pm.load_advice_rate() == {}
    assert fake_pm.load_observed("single:chess:user", limit=10) == []
    assert fake_pm.load_channel_stats("single:chess:user") == {}


# -- record_delivery_success: HIGH dishabituation ----------------------------


def test_delivery_success_high_dishabituates(fake_pm, cfg) -> None:
    # Pre-seed an almost-fully-habituated channel.
    fake_pm.save_habituation(
        "single:typing:user", {"h": 0.95, "last_event_ts": 50.0, "count": 9}
    )
    g = _gate()
    s = build_signature(SINGLE_HIGH_TYPING)  # high severity
    g.record_delivery_success(
        s, fake_pm, 100.0, decision=GateDecision.fire("n"), tier=2
    )
    hab = fake_pm.load_habituation("single:typing:user")
    # A high delivery DISHABITUATES — h reset toward 0 (spec §5 HIGH bypass).
    assert hab["h"] < 0.95
    assert hab["h"] <= 0.1


# -- record_delivery_success: audit_only (exempt) ----------------------------


def test_delivery_success_audit_only_minimal_write(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(EXEMPT_PAYLOAD)
    d = GateDecision.fire("exempt_high_correlated", deciding_arm="danger_exemption")
    g.record_delivery_success(s, fake_pm, 100.0, decision=d, tier=2, audit_only=True)

    # ONLY an audit emission entry — no h / channel_stats / observed.
    emissions = fake_pm.load_emissions(limit=10)
    assert len(emissions) == 1
    assert emissions[0]["audit_only"] is True

    assert fake_pm.load_habituation(s.state_key) == {}
    assert fake_pm.load_channel_stats(s.state_key) == {}
    assert fake_pm.load_observed(s.state_key, limit=10) == []
    assert fake_pm.load_advice_rate() == {}


def test_delivery_success_audit_only_emission_is_gating_invisible(fake_pm, cfg) -> None:
    # An audit-only emission must be ignored by the refractory arm (it is not a
    # real channel delivery) — read back as audit_only=True.
    g = _gate()
    s = build_signature(EXEMPT_PAYLOAD)
    d = GateDecision.fire("exempt_high_correlated")
    g.record_delivery_success(s, fake_pm, 100.0, decision=d, tier=2, audit_only=True)
    e = fake_pm.load_emissions(limit=10)[0]
    assert e["audit_only"] is True
    assert e.get("probe") is False


# -- record_suppression ------------------------------------------------------


def test_record_suppression_writes_authoritative_silence(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    d = GateDecision.suppress(
        "habituated",
        deciding_arm="habituation",
        metrics={"h_eff": 0.9},
        mrt_eligible=True,
        p_withhold=0.9,
    )
    ok = g.record_suppression(d, s, fake_pm, 100.0)
    assert ok is True

    silences = fake_pm.load_silence_records(limit=10)
    assert len(silences) == 1
    rec = silences[0]
    assert rec["decision_id"] == d.id
    assert rec["state_key"] == "single:chess:user"
    assert rec["arm"] == "habituation"
    assert rec["reason"] == "habituated"
    # IPW fields persisted on the silence (spec §6/§8 — offline IPW from records).
    assert rec["mrt_eligible"] is True
    assert rec["p_withhold"] == 0.9


def test_record_suppression_advances_reservoir_observed_channel_stats(
    fake_pm, cfg
) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    d = GateDecision.suppress("single_channel_insufficient", deciding_arm="reservoir")
    g.record_suppression(d, s, fake_pm, 100.0)

    # reservoir count advances (the evidence a suppressed event legitimately feeds)
    res = fake_pm.load_reservoir("single:chess:user")
    assert res["count"] >= 1.0
    assert res["last_ts"] == 100.0

    # observed-value window appended (novelty depends on it)
    obs = fake_pm.load_observed("single:chess:user", limit=10)
    assert len(obs) == 1
    assert obs[0]["value"] == 2.0

    # channel_stats: first suppression of a streak sets the streak timestamp.
    stats = fake_pm.load_channel_stats("single:chess:user")
    assert stats["consecutive_suppressions"] == 1
    assert stats["suppression_streak_started_ts"] == 100.0


def test_record_suppression_streak_started_only_on_first(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    d = GateDecision.suppress("habituated", deciding_arm="habituation")
    g.record_suppression(d, s, fake_pm, 100.0)
    g.record_suppression(d, s, fake_pm, 150.0)

    stats = fake_pm.load_channel_stats("single:chess:user")
    assert stats["consecutive_suppressions"] == 2
    # streak start stays at the FIRST suppression, not the latest.
    assert stats["suppression_streak_started_ts"] == 100.0


def test_record_suppression_returns_false_when_silence_write_fails(
    fake_pm, cfg, monkeypatch
) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    d = GateDecision.suppress("habituated", deciding_arm="habituation")

    def _boom(_record):
        raise RuntimeError("redis down")

    monkeypatch.setattr(fake_pm, "save_silence_record", _boom)
    assert g.record_suppression(d, s, fake_pm, 100.0) is False


# -- record_busy_skip --------------------------------------------------------


def test_record_busy_skip_bumps_channel_stats(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    tracked = g.record_busy_skip(s, fake_pm, 100.0)
    assert tracked is True

    stats = fake_pm.load_channel_stats("single:chess:user")
    assert stats["consecutive_suppressions"] == 1
    assert stats["suppression_streak_started_ts"] == 100.0

    # A best-effort delivery_failure is also written.
    failures = fake_pm.load_delivery_failures(limit=10)
    assert len(failures) == 1
    assert failures[0]["reason"] == "advisor_busy_skipped"


def test_record_busy_skip_untrackable_returns_false(fake_pm_at_cap, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM_NEWKEY_THAT_WOULD_SUPPRESS)
    # channel_stats hash is at cap; a brand-new key cannot be tracked.
    assert g.record_busy_skip(s, fake_pm_at_cap, 100.0) is False


# -- still_starved -----------------------------------------------------------


def test_still_starved_true_when_count_bound_passed(fake_pm, cfg) -> None:
    fake_pm.save_channel_stats(
        "single:chess:user",
        {"consecutive_suppressions": 8, "suppression_streak_started_ts": 10.0},
    )
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    assert g.still_starved(s, fake_pm, 100.0) is True


def test_still_starved_false_when_not_starved(fake_pm, cfg) -> None:
    fake_pm.save_channel_stats(
        "single:chess:user",
        {"consecutive_suppressions": 1, "suppression_streak_started_ts": 99.0},
    )
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)
    assert g.still_starved(s, fake_pm, 100.0) is False


def test_still_starved_safe_default_true_on_read_error(fake_pm, cfg) -> None:
    g = _gate()
    s = build_signature(SINGLE_MEDIUM)

    def _boom(_state_key):
        raise RuntimeError("redis read failed")

    fake_pm.load_channel_stats = _boom  # type: ignore[assignment]
    # safe default: assume starved → fire, never drop a release (spec §3/§4).
    assert g.still_starved(s, fake_pm, 100.0) is True
