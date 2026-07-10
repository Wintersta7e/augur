"""Conscientia config knobs — defaults, env coercion, validation."""

import pytest

from tabula.config import AugurConfig


def test_defaults():
    cfg = AugurConfig()
    assert cfg.conscientia_enabled is True
    assert cfg.conscientia_output_screen_enabled is True
    assert cfg.conscientia_regenerate_on_violation is True
    assert cfg.conscientia_regenerate_max == 1
    assert cfg.conscientia_teach_screen_enabled is True
    assert cfg.conscientia_inject_screen_enabled is True
    assert cfg.conscientia_proposal_screen_enabled is True
    assert cfg.conscientia_output_extra_patterns == ()
    assert cfg.conscientia_teach_extra_patterns == ()


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AUGUR_CONSCIENTIA_ENABLED", "false")
    monkeypatch.setenv("AUGUR_CONSCIENTIA_REGENERATE_MAX", "2")
    cfg = AugurConfig.from_env()
    assert cfg.conscientia_enabled is False
    assert cfg.conscientia_regenerate_max == 2


def test_extra_patterns_env_comma_split(monkeypatch):
    monkeypatch.setenv("AUGUR_CONSCIENTIA_OUTPUT_EXTRA_PATTERNS", "foo bar, baz ,, qux")
    monkeypatch.setenv("AUGUR_CONSCIENTIA_TEACH_EXTRA_PATTERNS", " solo ")
    cfg = AugurConfig.from_env()
    # comma-split, stripped, empties dropped — NOT per-character (the
    # auto-build loop's tuple(v) would char-split without the coercion)
    assert cfg.conscientia_output_extra_patterns == ("foo bar", "baz", "qux")
    assert cfg.conscientia_teach_extra_patterns == ("solo",)


def test_regenerate_max_bounds():
    with pytest.raises(ValueError):
        AugurConfig(conscientia_regenerate_max=-1)
    with pytest.raises(ValueError):
        AugurConfig(conscientia_regenerate_max=3)
    AugurConfig(conscientia_regenerate_max=0)  # bounds are [0, 2]
    AugurConfig(conscientia_regenerate_max=2)
