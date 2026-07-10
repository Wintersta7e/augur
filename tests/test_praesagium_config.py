"""Praesagium config knobs — defaults, env coercion, validation."""

import pytest
from tabula.config import AugurConfig


def test_defaults():
    cfg = AugurConfig()
    assert cfg.praesagium_enabled is True
    assert cfg.praesagium_emit_enabled is False  # watch-first
    assert cfg.praesagium_episode_cap_per_session == 2000
    assert cfg.praesagium_support_min_sessions == 3
    assert cfg.praesagium_conf_lower_min == 0.4
    assert cfg.praesagium_lift_min == 1.5
    assert cfg.praesagium_lag_min_s == 10.0
    assert cfg.praesagium_lag_max_s == 900.0
    assert cfg.praesagium_window_margin == 1.25
    assert cfg.praesagium_lag_stability_ratio == 2.0
    assert cfg.praesagium_max_patterns == 50
    assert cfg.praesagium_pattern_cooldown_s == 600.0
    assert cfg.praesagium_hit_rate_alpha == 0.2
    assert cfg.praesagium_hit_rate_retire_below == 0.3
    assert cfg.praesagium_retire_min_resolutions == 5
    assert cfg.praesagium_mine_max_sessions == 30
    assert cfg.praesagium_mine_min_interval_s == 1800.0
    assert cfg.praesagium_expiry_grace_s == 5.0
    assert cfg.praesagium_open_predictions_cap == 100
    assert cfg.praesagium_predictions_cap == 500


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AUGUR_PRAESAGIUM_EMIT_ENABLED", "true")
    monkeypatch.setenv("AUGUR_PRAESAGIUM_LIFT_MIN", "2.0")
    cfg = AugurConfig.from_env()
    assert cfg.praesagium_emit_enabled is True
    assert cfg.praesagium_lift_min == 2.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"praesagium_hit_rate_alpha": 0.6},  # (0, 0.5]
        {"praesagium_hit_rate_alpha": 0.0},
        {"praesagium_conf_lower_min": 0.96},
        {
            "praesagium_lag_min_s": 250.0,
            "praesagium_lag_max_s": 200.0,
        },  # cross: lag_min < lag_max (isolated)
        {"praesagium_support_min_sessions": 20, "praesagium_mine_max_sessions": 10},
        {
            "praesagium_hit_rate_retire_below": 0.5,
            "praesagium_conf_lower_min": 0.4,
        },  # cross: retire < conf_lower
        {"praesagium_episode_cap_per_session": 50},
        {"praesagium_expiry_grace_s": 0.1},
    ],
)
def test_bounds_reject(kwargs):
    with pytest.raises(ValueError):
        AugurConfig(**kwargs)
