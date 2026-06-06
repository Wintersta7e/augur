"""Unit tests for reasoning.advisor_gate — GateDecision + build_signature.

Task 2.1: GateDecision (frozen dataclass), the fire/suppress/downgrade
constructors (id minted only when None), as_fire id-preservation, and
build_signature determinism per spec §5.
"""

from __future__ import annotations

import dataclasses

import pytest

from reasoning.advisor_gate import GateDecision, build_signature


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
