import pytest
from tabula.config import AugurConfig


def test_imperator_defaults():
    c = AugurConfig.from_env()
    assert c.imperator_enabled is True
    assert c.imperator_tick_s == 5.0
    assert c.imperator_baseline_trained_obs == 15


def test_imperator_env_override(monkeypatch):
    monkeypatch.setenv("AUGUR_IMPERATOR_TICK_S", "2.5")
    monkeypatch.setenv("AUGUR_IMPERATOR_ENABLED", "false")
    c = AugurConfig.from_env()
    assert c.imperator_tick_s == 2.5
    assert c.imperator_enabled is False


def test_imperator_tick_bounds():
    with pytest.raises(ValueError):
        AugurConfig(imperator_tick_s=0.0)
