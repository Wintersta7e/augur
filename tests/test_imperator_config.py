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


def test_imperator_ii_defaults():
    c = AugurConfig.from_env()
    assert c.imperator_ii_enabled is True
    assert c.imperator_ii_apply_enabled is False
    assert c.imperator_ii_num_predict == 512
    assert c.min_prompt_len == 20


def test_imperator_ii_apply_env_override(monkeypatch):
    monkeypatch.setenv("AUGUR_IMPERATOR_II_APPLY_ENABLED", "true")
    assert AugurConfig.from_env().imperator_ii_apply_enabled is True


def test_imperator_ii_bounds():
    import pytest

    with pytest.raises(ValueError):
        AugurConfig(imperator_ii_num_predict=0)
