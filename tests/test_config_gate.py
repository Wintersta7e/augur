import logging

import pytest

from tabula.config import AugurConfig


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
    """Invalid env value is rejected by the coercion function, triggering a
    WARNING from augur.config and leaving the field at its default "note"."""
    monkeypatch.setenv("AUGUR_GATE_TIER1_MODE", "garbage")
    with caplog.at_level(logging.WARNING, logger="augur.config"):
        result = AugurConfig.from_env()
    assert result.gate_tier1_mode == "note"
    assert any(
        "AUGUR_GATE_TIER1_MODE" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    ), (
        "Expected a WARNING from augur.config about the invalid AUGUR_GATE_TIER1_MODE value"
    )


def test_gate_tier1_mode_invalid_direct_construction():
    """Direct construction with an invalid mode raises ValueError immediately,
    consistent with all other validated fields in __post_init__."""
    with pytest.raises(ValueError, match="gate_tier1_mode"):
        AugurConfig(gate_tier1_mode="garbage")


def test_behavioral_weight_bound_validated():
    with pytest.raises(ValueError):
        AugurConfig(gate_behavioral_weight=2.0, gate_explicit_weight=1.0)
