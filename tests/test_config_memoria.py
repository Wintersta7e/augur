"""Memoria (Lane 2) config fields: defaults, bounds, env coercion."""

import pytest

from tabula.config import AugurConfig


def test_memory_defaults():
    c = AugurConfig()
    assert c.memory_store_enabled is True
    assert c.memory_prune_r == 0.05
    assert c.memory_promote_s == 14
    assert c.memory_s_growth_factor == 0.5
    assert c.memory_s_min == 0.1
    assert c.memory_s_max == 365
    assert c.max_memory_items == 5000
    assert c.memory_decay_form == "exponential"


@pytest.mark.parametrize(
    "field,value",
    [
        ("memory_prune_r", 0.6),  # > 0.5
        ("memory_prune_r", -0.1),  # < 0
        ("memory_promote_s", 1),  # < 2
        ("memory_s_growth_factor", 6.0),  # > 5
        ("memory_s_min", 0.0),  # not in (0, 1]
        ("memory_s_min", 1.5),  # > 1
        ("memory_s_max", 5),  # < 10
        ("max_memory_items", 50),  # < 100
    ],
)
def test_memory_bounds_raise(field, value):
    with pytest.raises(ValueError):
        AugurConfig(**{field: value})


def test_memory_decay_form_enum_raises():
    with pytest.raises(ValueError):
        AugurConfig(memory_decay_form="linear")


def test_memory_env_override(monkeypatch):
    monkeypatch.setenv("AUGUR_MEMORY_STORE_ENABLED", "false")
    monkeypatch.setenv("AUGUR_MEMORY_PROMOTE_S", "20")
    monkeypatch.setenv("AUGUR_MAX_MEMORY_ITEMS", "1000")
    c = AugurConfig.from_env()
    assert c.memory_store_enabled is False  # _coerce_bool, not bool("false")
    assert c.memory_promote_s == 20
    assert c.max_memory_items == 1000
