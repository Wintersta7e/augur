"""Unit tests for the gate suppressor arms (spec §5).

Each arm is a pure private method ``_arm_<name>(self, sig, state, config, now,
rng) -> GateDecision | None``.  These tests drive arms through the real
``Gate.evaluate`` pipeline so arm ordering + the non-exempt-HIGH bypass are
exercised together (spec §5).
"""

from __future__ import annotations

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
    # Only a probe emission within the absolute window → must NOT suppress.
    fake_pm.save_emission(_emit("single:chess:user", "medium", ts=119.0, probe=True))
    fake_pm.save_emission(_emit("single:chess:x", "medium", ts=118.0, audit_only=True))
    g = Gate()
    d = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=120.0)
    assert d.action == "fire"


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
    # 0.5 * 1.0 = 0.5 < medium 1.0 → pressure passes → fire.
    d = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=10_000.0)
    assert d.action == "fire"


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
    assert d.action == "fire"


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
