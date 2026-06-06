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
    from reasoning.advisor_gate import Gate

    # A medium that trips no suppressor reaches the terminal passed_all_arms
    # fire.  The reservoir arm (Arm 5) holds a fresh single+medium until it has
    # accumulated evidence, so latch this channel committed (suppressing=False,
    # leaked count above OFF) to clear it — all other arms see unseen state.
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
