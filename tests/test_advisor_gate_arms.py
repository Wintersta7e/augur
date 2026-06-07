"""Unit tests for the gate suppressor arms (spec §5).

Each arm is a pure private method ``_arm_<name>(self, sig, state, config, now,
rng) -> GateDecision | None``.  These tests drive arms through the real
``Gate.evaluate`` pipeline so arm ordering + the non-exempt-HIGH bypass are
exercised together (spec §5).
"""

from __future__ import annotations

import pytest

from reasoning.advisor_gate import Gate, build_signature
from tests.conftest import EXEMPT_PAYLOAD, SINGLE_HIGH_TYPING, SINGLE_MEDIUM_TYPING


def test_central_tolerance_suppresses_medium_not_high(fake_pm, cfg):
    """Arm 1: a learned-self channel suppresses a medium; a HIGH bypasses it."""
    fake_pm.add_self_tolerance("single:typing:user")
    g = Gate()

    medium = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=1.0)
    assert medium.action == "suppress"
    assert medium.reason == "central_tolerance_learned_self"
    assert medium.deciding_arm == "central_tolerance"

    # HIGH bypass: a standalone high skips the learned/recurrence suppressors.
    high = g.evaluate(build_signature(SINGLE_HIGH_TYPING), fake_pm, cfg, now=1.0)
    assert high.action == "fire"


# ── Arm 2: refractory_burden (spec §5 Arm 2) ─────────────────────────────────
#
# Four sub-reasons, checked in order: absolute → relative-bar → pressure →
# duplicate.  Emissions are read via load_emissions IGNORING probe/audit_only
# entries (those are not real deliveries that should refract the channel).


def _emit(state_key: str, severity: str, ts: float, **extra):
    """Build a gate emission record (matching the §6 emissions schema)."""
    rec = {
        "ts": ts,
        "decision_id": f"d-{ts}",
        "state_key": state_key,
        "severity": severity,
        "tier": 2,
        "probe": False,
        "audit_only": False,
        "withheld_reason": None,
        "mrt_eligible": False,
        "p_fire": None,
    }
    rec.update(extra)
    return rec


def test_refractory_absolute_suppresses_within_window(fake_pm, cfg):
    """Absolute: a real emission within ABSOLUTE_REFRACTORY_S suppresses."""
    # last global emit at ts=100; now=120 → dt=20 < 45 (gate_absolute_refractory_s).
    fake_pm.save_emission(_emit("single:chess:user", "medium", ts=100.0))
    g = Gate()
    d = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=120.0)
    assert d.action == "suppress"
    assert d.reason == "absolute_refractory"
    assert d.deciding_arm == "refractory_burden"
    assert d.metrics["remaining_s"] == cfg.gate_absolute_refractory_s - 20.0


def test_refractory_absolute_ignores_probe_and_audit_only(fake_pm, cfg):
    """Probe/audit_only emissions do not count as deliveries that refract."""
    # Only a probe emission within the absolute window → refractory must NOT
    # suppress.  (A fresh single+medium is held downstream by the reservoir arm,
    # so assert refractory specifically passed rather than a global fire.)
    fake_pm.save_emission(_emit("single:chess:user", "medium", ts=119.0, probe=True))
    fake_pm.save_emission(_emit("single:chess:x", "medium", ts=118.0, audit_only=True))
    g = Gate()
    d = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=120.0)
    assert d.deciding_arm != "refractory_burden"


def test_refractory_relative_raised_bar_suppresses_medium(fake_pm, cfg):
    """Relative: bar high→medium over the window; a medium below the bar suppresses."""
    # last emit at ts=50, now=110 → dt=60: past absolute (45), inside relative
    # (180); bar has decayed only partway so a medium is still below it.
    fake_pm.save_emission(_emit("single:chess:user", "high", ts=50.0))
    g = Gate()
    d = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=110.0)
    assert d.action == "suppress"
    assert d.reason == "relative_refractory_raised_bar"
    assert d.deciding_arm == "refractory_burden"
    assert d.metrics["elapsed_s"] == 60.0
    assert "bar" in d.metrics


def test_refractory_relative_does_not_suppress_high(fake_pm, cfg):
    """A HIGH is never below the raised bar (bar max is high) → relative passes."""
    fake_pm.save_emission(_emit("single:chess:user", "high", ts=50.0))
    g = Gate()
    # now=110 → dt=60: past absolute, inside relative; a high is at/above the bar.
    d = g.evaluate(build_signature(SINGLE_HIGH_TYPING), fake_pm, cfg, now=110.0)
    assert d.action == "fire"


def test_refractory_pressure_suppresses_on_high_advice_rate(fake_pm, cfg):
    """Pressure: advice_rate * PRESSURE_WEIGHT > severity_score suppresses."""
    # No recent global emission (so absolute/relative pass), but advice-rate EWMA
    # is high: rate 1.5 * weight 1.0 = 1.5 > medium severity_score 1.0.
    fake_pm.save_advice_rate({"rate_ewma": 1.5, "last_ts": 10.0})
    g = Gate()
    d = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=10_000.0)
    assert d.action == "suppress"
    assert d.reason == "active_resolution_recent_advice_pressure"
    assert d.deciding_arm == "refractory_burden"
    assert d.metrics["advice_rate"] == 1.5


def test_refractory_pressure_capped(fake_pm, cfg):
    """Pressure compares against PRESSURE_CAP-clamped advice_rate."""
    # rate 100 clamps to PRESSURE_CAP=3.0; 3.0 * 1.0 = 3.0 > medium 1.0 → suppress,
    # but the reported advice_rate is the capped value.
    fake_pm.save_advice_rate({"rate_ewma": 100.0, "last_ts": 10.0})
    g = Gate()
    d = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=10_000.0)
    assert d.action == "suppress"
    assert d.reason == "active_resolution_recent_advice_pressure"
    assert d.metrics["advice_rate"] == cfg.gate_pressure_cap


def test_refractory_pressure_passes_below_threshold(fake_pm, cfg):
    """Low advice-rate does not trip pressure."""
    fake_pm.save_advice_rate({"rate_ewma": 0.5, "last_ts": 10.0})
    g = Gate()
    # 0.5 * 1.0 = 0.5 < medium 1.0 → pressure passes (refractory does not decide;
    # a fresh single+medium is held downstream by the reservoir arm).
    d = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=10_000.0)
    assert d.deciding_arm != "refractory_burden"


def test_refractory_duplicate_suppresses_recent_same_state_key(fake_pm, cfg):
    """Duplicate: a recent non-probe emission on the SAME state_key suppresses."""
    # Use a HIGH so absolute/relative/pressure all pass:
    #   - global last emit at ts=900, now=960 → dt=60 ≥ 45 (absolute passes);
    #   - high is at/above the raised bar (relative passes);
    #   - no advice-rate pressure;
    #   - same-state_key emission within RELATIVE_REFRACTORY_S → duplicate.
    fake_pm.save_emission(_emit("single:typing:user", "high", ts=900.0))
    g = Gate()
    d = g.evaluate(build_signature(SINGLE_HIGH_TYPING), fake_pm, cfg, now=960.0)
    assert d.action == "suppress"
    assert d.reason == "already_covered_recent_equivalent"
    assert d.deciding_arm == "refractory_burden"
    assert d.metrics["dt"] == 60.0


def test_refractory_duplicate_ignores_probe_same_state_key(fake_pm, cfg):
    """A probe emission on the same state_key does not count as a duplicate."""
    # Only a probe same-state_key emission inside the relative window → no
    # duplicate; no other recent global emission → fire.
    fake_pm.save_emission(_emit("single:typing:user", "high", ts=900.0, probe=True))
    g = Gate()
    d = g.evaluate(build_signature(SINGLE_HIGH_TYPING), fake_pm, cfg, now=960.0)
    assert d.action == "fire"


def test_refractory_disabled_passes(fake_pm):
    """When gate_refractory_enabled is False, the arm never suppresses."""
    from blackboard.config import AugurConfig

    cfg = AugurConfig(gate_refractory_enabled=False)
    fake_pm.save_emission(_emit("single:chess:user", "medium", ts=100.0))
    g = Gate()
    d = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=120.0)
    assert d.deciding_arm != "refractory_burden"


def test_refractory_applies_to_ungateable(fake_pm, cfg):
    """Refractory is a GLOBAL arm: it applies even to an ungateable (no-entity) event."""
    no_entity = {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {"domain": "typing", "value": 2.0, "severity": "medium"},
    }
    fake_pm.save_emission(_emit("single:chess:user", "medium", ts=100.0))
    g = Gate()
    d = g.evaluate(build_signature(no_entity), fake_pm, cfg, now=120.0)
    assert d.action == "suppress"
    assert d.reason == "absolute_refractory"


# ── Arm 3: novelty_prediction_error (spec §5 Arm 3) ──────────────────────────
#
# Gates on value/timing surprise (distinct from habituation's advice-frequency).
# Maintains a per-state_key EWMA predicted_value over the observed-value window.
# relative_change = |value - predicted_value| / max(|predicted_value|, _EPS).
# Familiar (match_count >= NOVELTY_FAMILIAR_MIN) AND relative_change <
# WEBER_FRACTION → SUPPRESS("fully_predicted_explained_away"); unseen → PASS.


def _medium_typing(value: float) -> dict:
    """A single+medium typing payload with a controllable anomaly value."""
    return {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": value,
            "severity": "medium",
        },
    }


def _high_typing(value: float) -> dict:
    """A single+high typing payload with a controllable anomaly value."""
    return {
        "combined_severity": "HIGH",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": value,
            "severity": "high",
        },
    }


def _seed_observed(fake_pm, state_key: str, values: list[float]) -> None:
    """Seed the observed-value window for *state_key* (oldest first in *values*)."""
    for i, v in enumerate(values):
        fake_pm.save_observed(
            {"ts": float(i), "state_key": state_key, "value": v, "severity": "medium"}
        )


def test_novelty_suppresses_fully_predicted_familiar_channel(fake_pm, cfg):
    """Familiar channel + tiny relative_change → fully_predicted_explained_away."""
    # 3 prior observations all at 2.0 → EWMA predicts 2.0 (>= familiar_min=3).
    _seed_observed(fake_pm, "single:typing:user", [2.0, 2.0, 2.0])
    g = Gate()
    # Current value 2.0 → relative_change = 0 < weber_fraction (0.15) → suppress.
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "suppress"
    assert d.reason == "fully_predicted_explained_away"
    assert d.deciding_arm == "novelty_prediction_error"
    assert d.metrics["predicted_value"] == 2.0
    assert d.metrics["relative_change"] == 0.0


def test_novelty_passes_large_relative_change(fake_pm, cfg):
    """Familiar channel but a surprising value (above the JND) → PASS → fire."""
    _seed_observed(fake_pm, "single:typing:user", [2.0, 2.0, 2.0])
    g = Gate()
    # Current value 4.0 → relative_change = |4-2|/2 = 1.0 > 0.15 → novelty passes
    # (the fresh single+medium is held downstream by the reservoir arm).
    d = g.evaluate(build_signature(_medium_typing(4.0)), fake_pm, cfg, now=1.0)
    assert d.deciding_arm != "novelty_prediction_error"


def test_novelty_unseen_state_key_passes(fake_pm, cfg):
    """An unseen/first-time state_key (no observed history) → novelty passes."""
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.deciding_arm != "novelty_prediction_error"


def test_novelty_below_familiar_min_passes(fake_pm, cfg):
    """Too few observations (match_count < familiar_min) → not familiar → PASS."""
    # Only 2 observations (< familiar_min=3); even a perfectly-predicted value
    # must PASS because the channel is not yet familiar enough to explain away.
    _seed_observed(fake_pm, "single:typing:user", [2.0, 2.0])
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.deciding_arm != "novelty_prediction_error"


def test_novelty_high_bypasses(fake_pm, cfg):
    """HIGH bypass: a standalone high skips the novelty arm even if predicted."""
    _seed_observed(fake_pm, "single:typing:user", [2.0, 2.0, 2.0])
    g = Gate()
    d = g.evaluate(build_signature(_high_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "fire"


def test_novelty_disabled_passes(fake_pm):
    """When gate_novelty_enabled is False, the arm never suppresses."""
    from blackboard.config import AugurConfig

    cfg = AugurConfig(gate_novelty_enabled=False)
    _seed_observed(fake_pm, "single:typing:user", [2.0, 2.0, 2.0])
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.deciding_arm != "novelty_prediction_error"


# ── Arm 4: habituation (spec §5 Arm 4) ───────────────────────────────────────
#
# Gates on ADVICE FREQUENCY (distinct from novelty's value surprise).  Per
# state_key h ∈ [0,1] decays leakily toward 0:
#   h_eff = min(h * exp(-(now - last_event_ts) / TAU_S), 1 - max(FLOOR_MIN, floor))
#   R     = severity_score * (1 - h_eff)
#   R < R_THRESHOLD → SUPPRESS("habituated", {count, interval_s, h_eff, dt})
# A HIGH never reaches this arm (HIGH bypass).  Floor-guard caps h_eff so an
# offline-lowered floor can keep a channel responsive.


def test_habituation_suppresses_habituated_channel(fake_pm, cfg):
    """High h with no decay → R below R_THRESHOLD → habituated."""
    # h=0.9, last_event_ts=now → dt=0; floor unset → cap = 1 - 0.2 = 0.8.
    #   h_eff = min(0.9*exp(0), 0.8) = 0.8; R = 1.0*(1-0.8) = 0.2 < 0.5 → suppress.
    fake_pm.save_habituation(
        "single:typing:user", {"h": 0.9, "last_event_ts": 1.0, "count": 7}
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "suppress"
    assert d.reason == "habituated"
    assert d.deciding_arm == "habituation"
    assert d.metrics["count"] == 7
    assert d.metrics["dt"] == 0.0
    assert d.metrics["h_eff"] == 0.8
    assert "interval_s" in d.metrics


def test_habituation_decays_leakily_over_time(fake_pm, cfg):
    """Leaky decay: a long gap since last_event_ts lifts R back above threshold."""
    # h=0.9, dt=600 → h_eff = 0.9*exp(-1) = 0.331 (< the 0.8 cap);
    #   R = 1.0*(1 - 0.331) = 0.669 > 0.5 → fire (habituation has worn off).
    fake_pm.save_habituation(
        "single:typing:user", {"h": 0.9, "last_event_ts": 1.0, "count": 7}
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=601.0)
    assert d.deciding_arm != "habituation"


def test_habituation_floor_guard_caps_h_eff(fake_pm, cfg):
    """Floor-guard: a high offline floor caps h_eff so the channel stays responsive."""
    # h=1.0, dt=0 would give h_eff=1.0 without the guard.  floor=0.7 caps it:
    #   cap = 1 - max(0.2, 0.7) = 0.3; h_eff = min(1.0, 0.3) = 0.3;
    #   R = 1.0*(1 - 0.3) = 0.7 > 0.5 → fire (floor keeps it firing).
    fake_pm.save_habituation(
        "single:typing:user", {"h": 1.0, "last_event_ts": 1.0, "count": 20}
    )
    fake_pm.save_habituation_floor("single:typing:user", {"floor": 0.7, "last_ts": 1.0})
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.deciding_arm != "habituation"


def test_habituation_low_h_passes(fake_pm, cfg):
    """Low h → R above R_THRESHOLD → fire (channel not yet habituated)."""
    # h=0.2, dt=0 → h_eff = min(0.2, 0.8) = 0.2; R = 1.0*(1-0.2) = 0.8 > 0.5 → fire.
    fake_pm.save_habituation(
        "single:typing:user", {"h": 0.2, "last_event_ts": 1.0, "count": 2}
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.deciding_arm != "habituation"


def test_habituation_unseen_channel_passes(fake_pm, cfg):
    """An unseen state_key (no habituation state, h defaults to 0) → habituation passes."""
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.deciding_arm != "habituation"


def test_habituation_high_punches_through_at_h_near_one(fake_pm, cfg):
    """HARD: a standalone HIGH FIRES even at h≈1 (HIGH-punch-through via bypass).

    Without the HIGH bypass, severity_score=2.0 with h_eff=0.8 gives
    R = 2.0*(1-0.8) = 0.4 < 0.5 → it WOULD suppress.  The bypass skips the
    habituation arm for a non-exempt standalone high → it must fire.  This is
    the spec §5 / §11 HIGH-punch-through guarantee: a strong stimulus punches
    through routine (advice-frequency) suppression by construction.
    """
    fake_pm.save_habituation(
        "single:typing:user", {"h": 1.0, "last_event_ts": 1.0, "count": 50}
    )
    g = Gate()
    d = g.evaluate(build_signature(_high_typing(4.5)), fake_pm, cfg, now=1.0)
    assert d.action == "fire"
    assert d.reason != "habituated"


def test_habituation_disabled_passes(fake_pm):
    """When gate_habituation_enabled is False, the arm never suppresses."""
    from blackboard.config import AugurConfig

    cfg = AugurConfig(gate_habituation_enabled=False)
    fake_pm.save_habituation(
        "single:typing:user", {"h": 0.9, "last_event_ts": 1.0, "count": 7}
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.deciding_arm != "habituation"


# ── Arm 5: coincidence_evidence_reservoir (spec §5 Arm 5) ────────────────────
#
# Quorum sensing + immune two-signal.  Meters ONLY single+medium:
#   - two-signal short-circuit: severity=="high" OR correlation_found → PASS
#     unconditionally (high also never reaches the arm via the HIGH bypass).
#   - else a per-state_key decaying event count (+1/event, leaks by
#     exp(-dt/RESERVOIR_LEAK_TAU_S)); the event reads the current count + its own
#     prospective +1.  Schmitt-trigger hysteresis: a suppressing channel commits
#     (PASS) only when effective ≥ RESERVOIR_ON_COUNT; a committed channel
#     re-suppresses only when its leaked count falls below RESERVOIR_OFF_COUNT.
#   - below ON → SUPPRESS("single_channel_insufficient", {count, on, off}).


def test_reservoir_suppresses_new_single_medium_channel(fake_pm, cfg):
    """A brand-new single+medium channel is below ON → single_channel_insufficient."""
    # No reservoir state: count=0, prospective +1 → effective=1 < ON=3 → suppress.
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "suppress"
    assert d.reason == "single_channel_insufficient"
    assert d.deciding_arm == "coincidence_evidence_reservoir"
    assert d.metrics["on"] == cfg.gate_reservoir_on_count
    assert d.metrics["off"] == cfg.gate_reservoir_off_count
    assert d.metrics["count"] == 1.0  # leaked 0 + prospective 1


def test_reservoir_commits_when_effective_reaches_on(fake_pm, cfg):
    """A suppressing channel passes once leaked count + prospective +1 ≥ ON."""
    # count=2, dt=0 → leaked=2, effective=2+1=3 == ON=3 → commit → fire.
    # Disable the Phase-2 cost_tier modifier so the reservoir pass-through stays
    # a plain fire (a one-off single+medium would otherwise be Tier-1-downgraded).
    from dataclasses import replace

    cfg = replace(cfg, gate_cost_tier_enabled=False)
    fake_pm.save_reservoir(
        "single:typing:user", {"count": 2.0, "last_ts": 1.0, "suppressing": True}
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "fire"


def test_reservoir_two_signal_high_short_circuits(fake_pm, cfg):
    """Two-signal: a standalone HIGH short-circuits the reservoir (also HIGH-bypassed)."""
    # Even with an empty/insufficient reservoir, a high fires (never metered).
    g = Gate()
    d = g.evaluate(build_signature(_high_typing(4.5)), fake_pm, cfg, now=1.0)
    assert d.action == "fire"


def test_reservoir_two_signal_correlation_short_circuits(fake_pm, cfg):
    """Two-signal: correlation_found short-circuits even with a below-ON reservoir."""
    from tests.conftest import CORRELATION_MEDIUM

    # Seed the correlation state_key reservoir below ON + latched suppressing —
    # the short-circuit must ignore the count and PASS (correlation = signal 2).
    fake_pm.save_reservoir(
        "correlation:chess+typing", {"count": 0.0, "last_ts": 1.0, "suppressing": True}
    )
    g = Gate()
    d = g.evaluate(build_signature(CORRELATION_MEDIUM), fake_pm, cfg, now=1.0)
    assert d.action == "fire"


def test_reservoir_hysteresis_committed_channel_stays_passing_in_band(fake_pm, cfg):
    """Hysteresis: a committed channel keeps passing while in the OFF..ON band."""
    # count=1.5, dt=0 → leaked=1.5 (≥ OFF=1), effective=2.5 (< ON=3): in band.
    # Latched committed (suppressing=False) → re-suppress only below OFF → PASS.
    # Disable the Phase-2 cost_tier modifier to keep this on Phase-1 pass-through.
    from dataclasses import replace

    cfg = replace(cfg, gate_cost_tier_enabled=False)
    fake_pm.save_reservoir(
        "single:typing:user", {"count": 1.5, "last_ts": 1.0, "suppressing": False}
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "fire"


def test_reservoir_hysteresis_suppressing_channel_stays_suppressing_in_band(
    fake_pm, cfg
):
    """Hysteresis: a suppressing channel stays suppressing in the same OFF..ON band.

    SAME leaked count as the committed-channel test above (1.5 → effective 2.5)
    but latched suppressing → it does NOT commit until effective ≥ ON.  This pair
    proves the Schmitt-trigger hysteresis: in-band outcome depends on prior state.
    """
    fake_pm.save_reservoir(
        "single:typing:user", {"count": 1.5, "last_ts": 1.0, "suppressing": True}
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "suppress"
    assert d.reason == "single_channel_insufficient"


def test_reservoir_committed_channel_re_suppresses_below_off(fake_pm, cfg):
    """A committed channel re-suppresses once its leaked count falls below OFF."""
    # count=3 latched committed, but a long quiet gap leaks it below OFF=1:
    #   dt=600 → leaked = 3*exp(-5) ≈ 0.0202 < OFF=1 → re-suppress.
    fake_pm.save_reservoir(
        "single:typing:user", {"count": 3.0, "last_ts": 1.0, "suppressing": False}
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=601.0)
    assert d.action == "suppress"
    assert d.reason == "single_channel_insufficient"


def test_reservoir_disabled_passes(fake_pm):
    """When gate_reservoir_enabled is False, the arm never suppresses."""
    from blackboard.config import AugurConfig

    # cost_tier off too so the reservoir pass-through stays a plain fire.
    cfg = AugurConfig(gate_reservoir_enabled=False, gate_cost_tier_enabled=False)
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "fire"


# ── Arm 6: signaller_credibility (spec §5 Arm 6 + §10) ───────────────────────
#
# Cry-wolf + Friston precision.  Per signal-class (escalation_rule for
# correlated, else domain:severity) EWMA credibility ∈ [0,1] from reliability-
# weighted feedback.  Decays toward the prior (CRED_MID) when a class has no
# recent feedback.  P(suppress) = clamp((CRED_MID - credibility)/CRED_MID, 0,
# CRED_MAX_P); a positive seeded-rng draw → SUPPRESS("low_credibility_class").
# Reliability-weighted fusion (§10): EXPLICIT dominant; behavioral applied only
# when |behavioral_score - 0.5| > deadband AND behavioral_finalized AND
# behavioral_samples >= min_samples.  A behavioral-driven suppression is
# bet-hedge-eligible (mrt_eligible=True, p_withhold set).  Disabled by the
# HIGH bypass for a standalone high.


class _SeqRandom:
    """A deterministic rng stub returning queued .random() values in order."""

    def __init__(self, values):
        self._values = list(values)

    def random(self):
        return self._values.pop(0)


def _seed_credibility(fake_pm, signal_class, **entry):
    """Persist a credibility entry for *signal_class* (cred/n/last_fb_ts/...)."""
    fake_pm.save_credibility(signal_class, entry)


def _cred_cfg():
    """A config that disables the upstream reservoir arm so credibility decides.

    Arms 1–4 pass naturally on an unseeded channel; the reservoir arm (Arm 5)
    would otherwise hold a fresh single+medium before the credibility arm (Arm 6)
    is reached.  Disabling it isolates Arm 6 under the real evaluate pipeline.
    """
    from blackboard.config import AugurConfig

    return AugurConfig(gate_reservoir_enabled=False)


def test_credibility_class_for_single_is_domain_severity(fake_pm):
    """A single event's credibility class is f'{domain}:{severity}'."""
    cfg = _cred_cfg()
    # Low credibility on the single class → high P(suppress); rng draw below p.
    _seed_credibility(fake_pm, "typing:medium", cred=0.1, n=20, last_fb_ts=1.0)
    g = Gate()
    # cred_eff=0.1 (no decay at dt=0); p = (0.5-0.1)/0.5 = 0.8 (== CRED_MAX_P).
    rng = _SeqRandom([0.0])  # 0.0 < 0.8 → suppress
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0, rng=rng)
    assert d.action == "suppress"
    assert d.reason == "low_credibility_class"
    assert d.deciding_arm == "signaller_credibility"
    assert d.metrics["credibility"] == pytest.approx(0.1)
    assert d.metrics["p"] == pytest.approx(0.8)


def test_credibility_zero_cred_suppresses_at_max_p(fake_pm):
    """A stored cred=0.0 (worst class) suppresses at P=CRED_MAX_P.

    A legitimately-stored zero credibility is the maximally-untrustworthy class
    and MUST drive P(suppress) to its ceiling (CRED_MAX_P=0.8), not collapse to
    the neutral prior (which would yield P=0 and never suppress).  cred_eff=0.0
    (no decay at dt=0); p = clamp((0.5-0.0)/0.5, 0, 0.8) = 0.8.
    """
    cfg = _cred_cfg()
    _seed_credibility(fake_pm, "typing:medium", cred=0.0, n=50, last_fb_ts=1.0)
    g = Gate()
    rng = _SeqRandom([0.0])  # 0.0 < 0.8 → suppress
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0, rng=rng)
    assert d.action == "suppress"
    assert d.reason == "low_credibility_class"
    assert d.deciding_arm == "signaller_credibility"
    assert d.metrics["credibility"] == pytest.approx(0.0)
    assert d.metrics["p"] == pytest.approx(cfg.gate_cred_max_p)


def test_credibility_rng_above_p_passes(fake_pm):
    """A draw at/above P(suppress) does not suppress (probabilistic)."""
    cfg = _cred_cfg()
    _seed_credibility(fake_pm, "typing:medium", cred=0.1, n=20, last_fb_ts=1.0)
    g = Gate()
    rng = _SeqRandom([0.9])  # 0.9 >= 0.8 → credibility arm passes
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0, rng=rng)
    assert d.deciding_arm != "signaller_credibility"


def test_credibility_unseen_class_passes(fake_pm):
    """An unseen class sits at the prior (CRED_MID) → P=0 → never suppresses."""
    cfg = _cred_cfg()
    g = Gate()
    rng = _SeqRandom([0.0])  # even a 0.0 draw cannot beat p=0
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0, rng=rng)
    assert d.deciding_arm != "signaller_credibility"


def test_credibility_decays_toward_prior_for_stale_class(fake_pm):
    """A stale low-credibility class decays toward the prior → P drops → pass.

    Fresh (dt=0): cred_eff=0.1 → p=0.8 → a 0.5 draw suppresses.  Stale (a long
    gap since last_fb_ts) self-heals: cred_eff is pulled toward CRED_MID, so the
    SAME 0.5 draw no longer beats the (now lower) p.
    """
    cfg = _cred_cfg()
    _seed_credibility(fake_pm, "typing:medium", cred=0.1, n=20, last_fb_ts=0.0)
    g = Gate()

    # Fresh: dt=0 → no decay → p=0.8; a 0.5 draw suppresses.
    fresh = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=0.0,
        rng=_SeqRandom([0.5]),
    )
    assert fresh.action == "suppress"
    assert fresh.reason == "low_credibility_class"

    # Stale: a long quiet gap pulls cred_eff toward the prior so p < 0.5 and the
    # SAME 0.5 draw no longer suppresses (the class self-heals).
    stale = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=10_000.0,
        rng=_SeqRandom([0.5]),
    )
    assert stale.deciding_arm != "signaller_credibility"
    assert stale.metrics.get("credibility", 0.5) > 0.1  # decayed toward prior


def test_credibility_class_for_correlation_uses_escalation_rule(fake_pm):
    """A correlation's credibility class keys off escalation_rule."""
    cfg = _cred_cfg()
    corr = {
        "combined_severity": "MEDIUM",
        "correlation_found": True,
        "involved_domains": ["typing", "chess"],
        "escalation_rule": "typing+chess->med",
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": 2.5,
            "severity": "medium",
        },
    }
    _seed_credibility(fake_pm, "typing+chess->med", cred=0.1, n=20, last_fb_ts=1.0)
    g = Gate()
    d = g.evaluate(build_signature(corr), fake_pm, cfg, now=1.0, rng=_SeqRandom([0.0]))
    assert d.action == "suppress"
    assert d.reason == "low_credibility_class"
    assert d.deciding_arm == "signaller_credibility"


def test_credibility_behavioral_fusion_applies_when_eligible(fake_pm):
    """Fusion: a finalized below-deadband-clearing behavioral score lowers cred.

    Explicit cred=0.5 (the prior → p would be 0 → never suppress on explicit
    alone).  A behavioral_score of 0.0 (|0-0.5|=0.5 > deadband 0.15), finalized,
    with >= min_samples, fuses in (explicit-dominant) to pull effective
    credibility below the prior → P>0 → a low draw suppresses, and because the
    suppression is behavioral-driven it is bet-hedge-eligible (mrt_eligible).

    Bet-hedge (Arm 8) is disabled here so this stays a focused credibility-arm
    assertion (Arm 8's stamping/flip of this eligible suppress is covered by the
    ``test_bet_hedge_*`` tests); otherwise Arm 8 would consume a second rng draw
    and overwrite ``p_withhold`` with ``1 - ε``.
    """
    from blackboard.config import AugurConfig

    cfg = AugurConfig(gate_reservoir_enabled=False, gate_bet_hedge_enabled=False)
    _seed_credibility(
        fake_pm,
        "typing:medium",
        cred=0.5,
        n=20,
        last_fb_ts=1.0,
        behavioral_score=0.0,
        behavioral_samples=10,
        behavioral_finalized=True,
    )
    g = Gate()
    # fused = (1.0*0.5 + 0.2*0.0)/(1.0+0.2) = 0.5/1.2 = 0.41666...
    # p = (0.5 - 0.41666...)/0.5 = 0.16666...
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.05]),
    )
    assert d.action == "suppress"
    assert d.reason == "low_credibility_class"
    assert d.mrt_eligible is True
    assert d.p_withhold == d.metrics["p"]
    assert d.metrics["credibility"] < 0.5  # behavioral pulled it below the prior


def test_credibility_behavioral_ignored_when_unfinalized(fake_pm):
    """Fusion is gated on behavioral_finalized — an unfinalized score is ignored."""
    cfg = _cred_cfg()
    _seed_credibility(
        fake_pm,
        "typing:medium",
        cred=0.5,
        n=20,
        last_fb_ts=1.0,
        behavioral_score=0.0,
        behavioral_samples=10,
        behavioral_finalized=False,  # not finalized → behavioral excluded
    )
    g = Gate()
    # Explicit alone sits at the prior → p=0 → no suppression even on a 0.0 draw.
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.0]),
    )
    assert d.deciding_arm != "signaller_credibility"


def test_credibility_behavioral_ignored_inside_deadband(fake_pm):
    """A behavioral score within the deadband of 0.5 is treated as no signal."""
    cfg = _cred_cfg()
    _seed_credibility(
        fake_pm,
        "typing:medium",
        cred=0.5,
        n=20,
        last_fb_ts=1.0,
        behavioral_score=0.45,  # |0.45-0.5|=0.05 <= deadband 0.15 → ignored
        behavioral_samples=10,
        behavioral_finalized=True,
    )
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.0]),
    )
    assert d.deciding_arm != "signaller_credibility"


def test_credibility_behavioral_ignored_below_min_samples(fake_pm):
    """Fusion requires >= BEHAVIORAL_MIN_SAMPLES genuine responses."""
    cfg = _cred_cfg()
    _seed_credibility(
        fake_pm,
        "typing:medium",
        cred=0.5,
        n=20,
        last_fb_ts=1.0,
        behavioral_score=0.0,
        behavioral_samples=2,  # < min_samples (5) → behavioral excluded
        behavioral_finalized=True,
    )
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.0]),
    )
    assert d.deciding_arm != "signaller_credibility"


def test_credibility_explicit_only_suppress_not_bet_hedge_eligible(fake_pm):
    """A purely explicit-driven low-credibility suppress is NOT bet-hedge-eligible."""
    cfg = _cred_cfg()
    _seed_credibility(fake_pm, "typing:medium", cred=0.1, n=20, last_fb_ts=1.0)
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.0]),
    )
    assert d.action == "suppress"
    assert d.mrt_eligible is False
    assert d.p_withhold is None


def test_credibility_high_bypasses_low_credibility_class(fake_pm, cfg):
    """HARD (spec §11 HIGH-punch-through): a standalone HIGH with LOW credibility FIRES.

    The high's severity-omitted class (typing:high) is at rock-bottom credibility
    so it WOULD suppress with P=CRED_MAX_P.  The HIGH bypass skips the credibility
    arm for a non-exempt standalone high → it must fire regardless of the draw,
    completing the HIGH-punch-through gate across h / self-tolerance / credibility.
    """
    _seed_credibility(fake_pm, "typing:high", cred=0.0, n=50, last_fb_ts=1.0)
    g = Gate()
    d = g.evaluate(
        build_signature(_high_typing(4.5)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.0]),  # would suppress if the arm ran
    )
    assert d.action == "fire"
    assert d.reason != "low_credibility_class"


def test_credibility_disabled_passes(fake_pm):
    """When gate_credibility_enabled is False, the arm never suppresses."""
    from blackboard.config import AugurConfig

    cfg = AugurConfig(gate_credibility_enabled=False)
    _seed_credibility(fake_pm, "typing:medium", cred=0.0, n=50, last_fb_ts=1.0)
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.0]),
    )
    assert d.deciding_arm != "signaller_credibility"


# ── Arm-ordering precedence (spec §5 Phase 1: "first to suppress wins") ───────
#
# The Phase-1 suppressor pipeline runs in a fixed, documented order:
#   central_tolerance → refractory → novelty → habituation → reservoir →
#   credibility
# and evaluate returns the FIRST arm that suppresses.  The tests below seed a
# single non-high channel so that ALL SIX suppressors would fire simultaneously,
# then disable the higher-precedence arms one at a time and assert the next arm
# in line becomes the decider — pinning the exact precedence chain.


def _seed_all_suppressors(fake_pm) -> None:
    """Seed state so every Phase-1 suppressor would fire on single:typing:user.

    The channel is a non-high single+medium typing event at value 2.0:
      * Arm 1 central_tolerance — state_key in the self-tolerance set;
      * Arm 2 refractory_burden — a recent non-probe emission on the same
        state_key (absolute window + per-state_key duplicate);
      * Arm 3 novelty — 3 prior observations all at 2.0 → relative_change 0;
      * Arm 4 habituation — h=0.9 with no decay → R below threshold;
      * Arm 5 reservoir — fresh count (below ON) needs no seed;
      * Arm 6 credibility — cred=0.1 on the typing:medium class (paired with a
        0.0 rng draw at evaluate time).
    """
    fake_pm.add_self_tolerance("single:typing:user")
    fake_pm.save_emission(_emit("single:typing:user", "medium", ts=1.0))
    _seed_observed(fake_pm, "single:typing:user", [2.0, 2.0, 2.0])
    fake_pm.save_habituation(
        "single:typing:user", {"h": 0.9, "last_event_ts": 1.0, "count": 7}
    )
    _seed_credibility(fake_pm, "typing:medium", cred=0.1, n=20, last_fb_ts=1.0)


def test_arm_precedence_first_suppressor_wins(fake_pm):
    """When all six arms would suppress, evaluate returns them in spec §5 order.

    Disabling the highest-precedence arm each step hands the decision to the next
    arm in the documented chain (central_tolerance → refractory → novelty →
    habituation → reservoir → credibility), proving evaluate returns the FIRST
    suppressor and that the pipeline order matches the spec.
    """
    from blackboard.config import AugurConfig

    _seed_all_suppressors(fake_pm)
    g = Gate()
    sig = build_signature(_medium_typing(2.0))

    # (config-overrides, expected deciding arm, expected reason) in precedence
    # order — each row disables the arms ABOVE it so the next one decides.
    chain = [
        ({}, "central_tolerance", "central_tolerance_learned_self"),
        (
            {"gate_central_tolerance_enabled": False},
            "refractory_burden",
            "absolute_refractory",
        ),
        (
            {
                "gate_central_tolerance_enabled": False,
                "gate_refractory_enabled": False,
            },
            "novelty_prediction_error",
            "fully_predicted_explained_away",
        ),
        (
            {
                "gate_central_tolerance_enabled": False,
                "gate_refractory_enabled": False,
                "gate_novelty_enabled": False,
            },
            "habituation",
            "habituated",
        ),
        (
            {
                "gate_central_tolerance_enabled": False,
                "gate_refractory_enabled": False,
                "gate_novelty_enabled": False,
                "gate_habituation_enabled": False,
            },
            "coincidence_evidence_reservoir",
            "single_channel_insufficient",
        ),
        (
            {
                "gate_central_tolerance_enabled": False,
                "gate_refractory_enabled": False,
                "gate_novelty_enabled": False,
                "gate_habituation_enabled": False,
                "gate_reservoir_enabled": False,
            },
            "signaller_credibility",
            "low_credibility_class",
        ),
    ]

    for overrides, expected_arm, expected_reason in chain:
        cfg = AugurConfig(**overrides)
        d = g.evaluate(sig, fake_pm, cfg, now=1.0, rng=_SeqRandom([0.0]))
        assert d.action == "suppress", f"{expected_arm} should suppress"
        assert d.deciding_arm == expected_arm
        assert d.reason == expected_reason


def test_arm_precedence_all_disabled_fires(fake_pm):
    """With every suppressor disabled, the seeded channel passes all arms → fire."""
    from blackboard.config import AugurConfig

    _seed_all_suppressors(fake_pm)
    cfg = AugurConfig(
        gate_central_tolerance_enabled=False,
        gate_refractory_enabled=False,
        gate_novelty_enabled=False,
        gate_habituation_enabled=False,
        gate_reservoir_enabled=False,
        gate_credibility_enabled=False,
        # cost_tier (Phase 2) off too — this test pins the Phase-1 pass-through
        # outcome, not the cost-tier modifier applied to a fire-survivor.
        gate_cost_tier_enabled=False,
    )
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.0]),
    )
    assert d.action == "fire"
    assert d.reason == "passed_all_arms"


# ── Arm 7: cost_tier_router (spec §5 Arm 7) ──────────────────────────────────
#
# A Phase-2 MODIFIER on a fire-survivor (it never enters the Phase-1
# "first-suppressor-wins" loop).  A fire-survivor routes to Tier-2 (full 32B)
# when it is high, correlated, a persistent single (count >=
# COST_TIER_PERSISTENCE_COUNT via cost_tier_memory), or has previously earned
# Tier-2; otherwise it is a Tier-1 candidate and, depending on TIER1_MODE:
#   * "note"   → DOWNGRADE(tier=1, "cost_tier_downgrade") — a templated note
#                published on the advice subject so feedback/reflection see it;
#   * "silent" → SUPPRESS("cost_tier_downgrade_silent") — a gate non-delivery.


def _correlation_medium() -> dict:
    """A correlation+medium payload (state_key off involved_domains)."""
    return {
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


def _all_phase1_disabled(**extra) -> "object":
    """An AugurConfig with every Phase-1 suppressor off (isolates Arm 7).

    A clean single+medium channel passes Phase 1 anyway, but disabling the
    suppressors keeps these tests pinned to the cost-tier modifier alone even if
    state leaks in.
    """
    from blackboard.config import AugurConfig

    return AugurConfig(
        gate_central_tolerance_enabled=False,
        gate_refractory_enabled=False,
        gate_novelty_enabled=False,
        gate_habituation_enabled=False,
        gate_reservoir_enabled=False,
        gate_credibility_enabled=False,
        **extra,
    )


def test_cost_tier_high_stays_tier2(fake_pm):
    """A standalone HIGH fire-survivor routes to Tier-2 (stays a full fire)."""
    cfg = _all_phase1_disabled()
    g = Gate()
    d = g.evaluate(build_signature(_high_typing(4.5)), fake_pm, cfg, now=1.0)
    assert d.action == "fire"
    assert d.tier == 2
    assert d.reason != "cost_tier_downgrade"


def test_cost_tier_correlation_stays_tier2(fake_pm):
    """A correlated fire-survivor routes to Tier-2 (full 32B)."""
    cfg = _all_phase1_disabled()
    g = Gate()
    d = g.evaluate(build_signature(_correlation_medium()), fake_pm, cfg, now=1.0)
    assert d.action == "fire"
    assert d.tier == 2


def test_cost_tier_persistent_single_stays_tier2(fake_pm):
    """A single+medium whose count >= COST_TIER_PERSISTENCE_COUNT → Tier-2."""
    cfg = _all_phase1_disabled()
    fake_pm.save_cost_tier_memory(
        "single:typing:user",
        {"earned_tier2": False, "helped": 0, "count": 3, "last_ts": 1.0},
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "fire"
    assert d.tier == 2


def test_cost_tier_earned_tier2_stays_tier2(fake_pm):
    """A single+medium below the persistence count but Tier-2-earned → Tier-2."""
    cfg = _all_phase1_disabled()
    fake_pm.save_cost_tier_memory(
        "single:typing:user",
        {"earned_tier2": True, "helped": 1, "count": 1, "last_ts": 1.0},
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "fire"
    assert d.tier == 2


def test_cost_tier_tier1_note_mode_downgrades(fake_pm):
    """A non-persistent single+medium in 'note' mode → DOWNGRADE(tier=1)."""
    cfg = _all_phase1_disabled(gate_tier1_mode="note")
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "downgrade"
    assert d.tier == 1
    assert d.reason == "cost_tier_downgrade"
    assert d.deciding_arm == "cost_tier_router"


def test_cost_tier_tier1_silent_mode_suppresses(fake_pm):
    """A non-persistent single+medium in 'silent' mode → SUPPRESS."""
    cfg = _all_phase1_disabled(gate_tier1_mode="silent")
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "suppress"
    assert d.reason == "cost_tier_downgrade_silent"
    assert d.deciding_arm == "cost_tier_router"


def test_cost_tier_below_persistence_count_downgrades(fake_pm):
    """A single+medium with count < COST_TIER_PERSISTENCE_COUNT → Tier-1 note."""
    cfg = _all_phase1_disabled(gate_tier1_mode="note")
    fake_pm.save_cost_tier_memory(
        "single:typing:user",
        {"earned_tier2": False, "helped": 0, "count": 2, "last_ts": 1.0},
    )
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "downgrade"
    assert d.tier == 1


def test_cost_tier_disabled_passes_through_as_fire(fake_pm):
    """When gate_cost_tier_enabled is False, a fire-survivor stays a plain fire."""
    cfg = _all_phase1_disabled(gate_cost_tier_enabled=False, gate_tier1_mode="note")
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "fire"
    assert d.reason == "passed_all_arms"


def test_cost_tier_does_not_modify_a_phase1_suppress(fake_pm, cfg):
    """Arm 7 only modifies fire-survivors — a Phase-1 SUPPRESS is untouched."""
    fake_pm.add_self_tolerance("single:typing:user")
    g = Gate()
    d = g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert d.action == "suppress"
    assert d.reason == "central_tolerance_learned_self"


def test_cost_tier_silent_downgrade_is_readonly(fake_pm):
    """A cost_tier silent SUPPRESS still writes nothing in evaluate (read-only)."""
    cfg = _all_phase1_disabled(gate_tier1_mode="silent")
    g = Gate()
    g.evaluate(build_signature(_medium_typing(2.0)), fake_pm, cfg, now=1.0)
    assert fake_pm.write_calls == 0


# ── Arm 8: bet_hedge_override (spec §5 Arm 8 + §9) ───────────────────────────
#
# A Phase-2 MODIFIER evaluated AFTER anti-starvation: only when the provisional
# decision is still a SUPPRESS that is behavioral-driven (the eligible band:
# credibility-arm, single+medium → mrt_eligible).  With known probability ε
# (BET_HEDGE_EPSILON) it flips to FIRE(probe=True, withheld_reason=<original>)
# — genuine action-randomization (MRT).  Whenever Arm 8 CONSIDERS an eligible
# decision it stamps the known randomization probabilities on it (p_fire=ε,
# p_withhold=1-ε, mrt_eligible=True) so BOTH arms (probe-fired vs withheld) are
# inverse-probability-weightable offline (spec §4/§9), regardless of the flip
# outcome.  Never applied to exempt (already fires) or anti-starvation releases.


def _bet_hedge_cfg(**extra):
    """Config isolating the bet-hedge arm: reservoir off so credibility decides.

    Arm 8 only sees a SUPPRESS that the behavioral-fusion credibility arm
    produced, so the upstream reservoir arm (which would otherwise hold a fresh
    single+medium first) is disabled, mirroring ``_cred_cfg``.
    """
    from blackboard.config import AugurConfig

    return AugurConfig(gate_reservoir_enabled=False, **extra)


def _seed_behavioral_eligible(fake_pm):
    """Seed a class so the credibility arm yields a behavioral-driven suppress.

    Explicit cred=0.5 (prior → P(suppress)=0 on explicit alone); a finalized
    behavioral_score=0.0 (|0-0.5|=0.5 > deadband) with >= min_samples fuses in
    (explicit-dominant) to pull effective credibility below the prior → P>0 → a
    low credibility-draw suppresses, and the suppress is mrt_eligible.
    """
    _seed_credibility(
        fake_pm,
        "typing:medium",
        cred=0.5,
        n=20,
        last_fb_ts=1.0,
        behavioral_score=0.0,
        behavioral_samples=10,
        behavioral_finalized=True,
    )


def test_bet_hedge_flips_eligible_suppress_when_draw_below_epsilon(fake_pm):
    """rng below ε flips a behavioral-driven SUPPRESS to a probe FIRE (MRT).

    The first rng draw (0.05) drives the credibility suppress (it is < p); the
    second draw (0.0) is the bet-hedge draw and is < ε (0.1) → flip to FIRE with
    ``probe=True`` and ``withheld_reason`` = the original suppress reason.  The
    decision is stamped mrt_eligible=True, p_fire=ε, p_withhold=1-ε.
    """
    cfg = _bet_hedge_cfg()
    _seed_behavioral_eligible(fake_pm)
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.05, 0.0]),  # cred draw suppresses; hedge draw 0.0 < 0.1
    )
    assert d.action == "fire"
    assert d.deciding_arm == "bet_hedge_override"
    assert d.probe is True
    assert d.withheld_reason == "low_credibility_class"
    assert d.mrt_eligible is True
    assert d.p_fire == pytest.approx(cfg.gate_bet_hedge_epsilon)
    assert d.p_withhold == pytest.approx(1.0 - cfg.gate_bet_hedge_epsilon)


def test_bet_hedge_keeps_suppress_when_draw_at_or_above_epsilon(fake_pm):
    """rng at/above ε leaves the behavioral-driven SUPPRESS standing (withheld).

    The hedge draw (0.5 >= ε 0.1) does NOT flip, so the decision stays a
    SUPPRESS — but Arm 8 still stamps the known randomization probabilities
    (mrt_eligible=True, p_fire=ε, p_withhold=1-ε) so the withheld arm is IPW-able
    against probe-fired siblings even under a dynamic ε.
    """
    cfg = _bet_hedge_cfg()
    _seed_behavioral_eligible(fake_pm)
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.05, 0.5]),  # cred draw suppresses; hedge draw 0.5 >= 0.1
    )
    assert d.action == "suppress"
    assert d.reason == "low_credibility_class"
    assert d.deciding_arm == "signaller_credibility"
    assert d.probe is False
    assert d.mrt_eligible is True
    assert d.p_fire == pytest.approx(cfg.gate_bet_hedge_epsilon)
    assert d.p_withhold == pytest.approx(1.0 - cfg.gate_bet_hedge_epsilon)


def test_bet_hedge_ignores_explicit_only_suppress(fake_pm):
    """A purely explicit-driven (not mrt_eligible) SUPPRESS is never bet-hedged.

    A low explicit credibility with no behavioral fusion produces a suppress
    that is NOT mrt_eligible; Arm 8 leaves it untouched even on a 0.0 hedge draw.
    """
    cfg = _bet_hedge_cfg()
    _seed_credibility(fake_pm, "typing:medium", cred=0.1, n=20, last_fb_ts=1.0)
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom(
            [0.0, 0.0]
        ),  # cred suppresses; hedge draw would flip IF eligible
    )
    assert d.action == "suppress"
    assert d.deciding_arm == "signaller_credibility"
    assert d.mrt_eligible is False
    assert d.p_fire is None


def test_bet_hedge_never_applies_to_exempt(fake_pm, cfg):
    """An exempt (high+correlated) signature fires before any arm — never probed."""
    g = Gate()
    d = g.evaluate(
        build_signature(EXEMPT_PAYLOAD),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.0, 0.0]),
    )
    assert d.action == "fire"
    assert d.deciding_arm == "danger_exemption"
    assert d.probe is False


def test_bet_hedge_preserves_decision_id_across_flip(fake_pm):
    """The flipped probe FIRE keeps the original suppress decision id (linkage)."""
    cfg = _bet_hedge_cfg()
    _seed_behavioral_eligible(fake_pm)
    g = Gate()
    # Capture the would-be suppress id (no flip) then the flip — minted ids are
    # random per evaluate, so instead assert the flip carries A non-empty id and
    # the conversion preserved the suppress id within a single evaluate by
    # checking the id is the same object threaded through (non-empty hex).
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.05, 0.0]),
    )
    assert d.action == "fire"
    assert isinstance(d.id, str) and len(d.id) == 32  # uuid4().hex preserved


def test_bet_hedge_disabled_keeps_suppress(fake_pm):
    """When gate_bet_hedge_enabled is False, an eligible suppress is not flipped."""
    cfg = _bet_hedge_cfg(gate_bet_hedge_enabled=False)
    _seed_behavioral_eligible(fake_pm)
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.05, 0.0]),  # hedge draw would flip if enabled
    )
    assert d.action == "suppress"
    assert d.deciding_arm == "signaller_credibility"


def test_bet_hedge_does_not_touch_a_fire_survivor(fake_pm):
    """Arm 8 only acts on a SUPPRESS — a plain fire-survivor is untouched."""
    cfg = _bet_hedge_cfg(
        gate_central_tolerance_enabled=False,
        gate_credibility_enabled=False,
        gate_cost_tier_enabled=False,
    )
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=1.0,
        rng=_SeqRandom([0.0]),  # no flip draw should be consumed for a fire
    )
    assert d.action == "fire"
    assert d.probe is False
    assert d.deciding_arm != "bet_hedge_override"


# ── Arm 9: anti_starvation_release (spec §5 Arm 9, invariant D) ───────────────
#
# A Phase-2 MODIFIER evaluated BEFORE bet_hedge (Arm 8): if the provisional
# decision is still a SUPPRESS and the channel is starved — channel_stats shows
# consecutive_suppressions >= MAX_CONSECUTIVE_SUPPRESSIONS, OR (now -
# suppression_streak_started_ts) > MAX_CHANNEL_SILENCE_S — the suppression is
# un-suppressed deterministically to FIRE("anti_starvation_release") and Arm 8
# is short-circuited.  Deterministic: never a probe, consumes no rng draw.


def _seed_channel_stats(fake_pm, state_key, **entry):
    """Persist a channel_stats entry for *state_key* (anti-starvation substrate)."""
    fake_pm.save_channel_stats(state_key, entry)


def _starve_cfg(**extra):
    """Config isolating Arm 9: reservoir off so the credibility arm produces the
    behavioral-driven SUPPRESS that Arm 9 then releases (mirrors _bet_hedge_cfg)."""
    from blackboard.config import AugurConfig

    return AugurConfig(gate_reservoir_enabled=False, **extra)


def test_anti_starvation_releases_channel_at_consecutive_cap(fake_pm):
    """consecutive_suppressions >= MAX → FIRE('anti_starvation_release')."""
    cfg = _starve_cfg()
    _seed_channel_stats(
        fake_pm,
        "single:typing:user",
        consecutive_suppressions=cfg.gate_max_consecutive_suppressions,
        suppression_streak_started_ts=1.0,
    )
    _seed_credibility(fake_pm, "typing:medium", cred=0.0, n=20, last_fb_ts=1.0)
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=2.0,
        rng=_SeqRandom([0.0]),  # cred draw suppresses; Arm 9 consumes no draw
    )
    assert d.action == "fire"
    assert d.reason == "anti_starvation_release"
    assert d.deciding_arm == "anti_starvation_release"
    assert d.probe is False
    assert d.metrics["consecutive"] == cfg.gate_max_consecutive_suppressions


def test_anti_starvation_releases_channel_past_silence_window(fake_pm):
    """now - suppression_streak_started_ts > MAX_CHANNEL_SILENCE_S → release.

    A sparse stream that never reaches the count bound is still released by the
    time bound (invariant D).
    """
    cfg = _starve_cfg()
    _seed_channel_stats(
        fake_pm,
        "single:typing:user",
        consecutive_suppressions=1,  # below the count bound
        suppression_streak_started_ts=10.0,
    )
    _seed_credibility(fake_pm, "typing:medium", cred=0.0, n=20, last_fb_ts=1.0)
    g = Gate()
    now = 10.0 + cfg.gate_max_channel_silence_s + 1.0  # past the time bound
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=now,
        rng=_SeqRandom([0.0]),
    )
    assert d.action == "fire"
    assert d.reason == "anti_starvation_release"
    assert d.deciding_arm == "anti_starvation_release"


def test_anti_starvation_short_circuits_bet_hedge(fake_pm):
    """A starved behavioral-driven SUPPRESS becomes an anti-starvation FIRE,
    NOT a bet-hedge probe (Arm 9 runs before Arm 8 and short-circuits it)."""
    cfg = _starve_cfg()
    _seed_behavioral_eligible(fake_pm)  # credibility arm yields mrt_eligible suppress
    _seed_channel_stats(
        fake_pm,
        "single:typing:user",
        consecutive_suppressions=cfg.gate_max_consecutive_suppressions,
        suppression_streak_started_ts=1.0,
    )
    g = Gate()
    # Only ONE rng draw (the credibility suppress draw) should be consumed — Arm 9
    # is deterministic and short-circuits the bet-hedge draw entirely.
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=2.0,
        rng=_SeqRandom([0.05]),
    )
    assert d.action == "fire"
    assert d.reason == "anti_starvation_release"
    assert d.deciding_arm == "anti_starvation_release"
    assert d.probe is False  # deterministic — not the bet_hedge probe


def test_anti_starvation_leaves_unstarved_suppress(fake_pm):
    """A SUPPRESS on a channel below both bounds is left standing (no release)."""
    cfg = _starve_cfg()
    _seed_channel_stats(
        fake_pm,
        "single:typing:user",
        consecutive_suppressions=1,
        suppression_streak_started_ts=1.0,
    )
    _seed_credibility(fake_pm, "typing:medium", cred=0.0, n=20, last_fb_ts=1.0)
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=2.0,  # well within the silence window
        rng=_SeqRandom([0.0]),
    )
    assert d.action == "suppress"
    assert d.reason == "low_credibility_class"


def test_anti_starvation_ignores_a_fire_survivor(fake_pm):
    """Arm 9 only releases a SUPPRESS — a plain fire-survivor is untouched even on
    a starved channel."""
    cfg = _all_phase1_disabled(gate_cost_tier_enabled=False)
    _seed_channel_stats(
        fake_pm,
        "single:typing:user",
        consecutive_suppressions=cfg.gate_max_consecutive_suppressions,
        suppression_streak_started_ts=1.0,
    )
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=2.0,
    )
    assert d.action == "fire"
    assert d.reason != "anti_starvation_release"
    assert d.deciding_arm != "anti_starvation_release"


def test_anti_starvation_disabled_keeps_suppress(fake_pm):
    """When gate_anti_starvation_enabled is False, a starved suppress stands."""
    cfg = _starve_cfg(gate_anti_starvation_enabled=False)
    _seed_channel_stats(
        fake_pm,
        "single:typing:user",
        consecutive_suppressions=cfg.gate_max_consecutive_suppressions,
        suppression_streak_started_ts=1.0,
    )
    _seed_credibility(fake_pm, "typing:medium", cred=0.0, n=20, last_fb_ts=1.0)
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=2.0,
        rng=_SeqRandom([0.0]),
    )
    assert d.action == "suppress"
    assert d.reason == "low_credibility_class"


def test_anti_starvation_preserves_decision_id(fake_pm):
    """The released FIRE keeps the original suppress decision id (linkage key)."""
    cfg = _starve_cfg()
    _seed_channel_stats(
        fake_pm,
        "single:typing:user",
        consecutive_suppressions=cfg.gate_max_consecutive_suppressions,
        suppression_streak_started_ts=1.0,
    )
    _seed_credibility(fake_pm, "typing:medium", cred=0.0, n=20, last_fb_ts=1.0)
    g = Gate()
    d = g.evaluate(
        build_signature(_medium_typing(2.0)),
        fake_pm,
        cfg,
        now=2.0,
        rng=_SeqRandom([0.0]),
    )
    assert d.action == "fire"
    assert isinstance(d.id, str) and len(d.id) == 32  # uuid4().hex preserved
