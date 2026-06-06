"""Unit tests for the gate suppressor arms (spec §5).

Each arm is a pure private method ``_arm_<name>(self, sig, state, config, now,
rng) -> GateDecision | None``.  These tests drive arms through the real
``Gate.evaluate`` pipeline so arm ordering + the non-exempt-HIGH bypass are
exercised together (spec §5).
"""

from __future__ import annotations

import pytest

from reasoning.advisor_gate import Gate, build_signature
from tests.conftest import SINGLE_HIGH_TYPING, SINGLE_MEDIUM_TYPING


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

    cfg = AugurConfig(gate_reservoir_enabled=False)
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
    """
    cfg = _cred_cfg()
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
