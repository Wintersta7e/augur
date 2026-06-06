import pytest
from blackboard.config import AugurConfig


def test_gate_defaults_present():
    c = AugurConfig()
    assert c.gate_enabled is True
    assert c.gate_absolute_refractory_s == 45
    assert c.gate_tier1_mode == "note"
    assert c.gate_weber_fraction == 0.15
    assert 0 <= c.gate_behavioral_weight <= c.gate_explicit_weight


def test_gate_tier1_mode_env_override(monkeypatch):
    monkeypatch.setenv("AUGUR_GATE_TIER1_MODE", "silent")
    assert AugurConfig.from_env().gate_tier1_mode == "silent"


def test_gate_tier1_mode_garbage_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("AUGUR_GATE_TIER1_MODE", "garbage")
    assert (
        AugurConfig.from_env().gate_tier1_mode == "note"
    )  # invalid → default + warning


def test_behavioral_weight_bound_validated():
    with pytest.raises(ValueError):
        AugurConfig(gate_behavioral_weight=2.0, gate_explicit_weight=1.0)
