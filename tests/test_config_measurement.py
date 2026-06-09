"""Lane-1 measurement + data-quality config knobs (spec 2026-06-09)."""

import pytest

from blackboard.config import AugurConfig


def test_measurement_defaults():
    c = AugurConfig()
    assert c.post_decision_window == 3
    assert c.min_baseline_std == 0.01
    assert c.outcome_trend_bonus == 0.1
    assert c.gate_mrt_withheld_rating is False
    assert c.gate_mrt_withheld_rating_rate == 0.12
    assert c.gate_mrt_withheld_rating_max_sessions == 15
    assert c.drift_detector_enabled is True
    assert c.drift_detector == "adwin"
    assert c.drift_reset_cooldown_obs == 30
    assert c.drift_restart_std_factor == 1.0
    assert c.prompt_rollback_margin == 0.1
    assert isinstance(c.prompt_forbidden_patterns, tuple)
    assert "take a break" in [p.lower() for p in c.prompt_forbidden_patterns]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"post_decision_window": 1},
        {"post_decision_window": 51},
        {"min_baseline_std": 0.0},
        {"outcome_trend_bonus": 0.6},
        {"gate_mrt_withheld_rating_rate": 0.6},  # cap is 0.5
        {"gate_mrt_withheld_rating_max_sessions": 0},
        {"drift_detector": "bogus"},
        {"drift_reset_cooldown_obs": -1},
        {"drift_restart_std_factor": 0.0},  # below [0.25, 4.0]
        {"drift_restart_std_factor": 5.0},  # above [0.25, 4.0] (matches the clamp)
        {"prompt_rollback_margin": 1.5},
    ],
)
def test_measurement_bounds_rejected(kwargs):
    with pytest.raises(ValueError):
        AugurConfig(**kwargs)


def test_drift_detector_env_coercion(monkeypatch):
    monkeypatch.setenv("AUGUR_DRIFT_DETECTOR", "pagehinkley")
    monkeypatch.setenv("AUGUR_GATE_MRT_WITHHELD_RATING", "true")
    monkeypatch.setenv("AUGUR_GATE_MRT_WITHHELD_RATING_RATE", "0.2")
    c = AugurConfig.from_env()
    assert c.drift_detector == "pagehinkley"
    assert c.gate_mrt_withheld_rating is True
    assert c.gate_mrt_withheld_rating_rate == 0.2


def test_forbidden_patterns_env_comma_split(monkeypatch):
    monkeypatch.setenv("AUGUR_PROMPT_FORBIDDEN_PATTERNS", "foo bar, baz ,, qux")
    c = AugurConfig.from_env()
    # comma-split, stripped, empties dropped — NOT per-character
    assert c.prompt_forbidden_patterns == ("foo bar", "baz", "qux")
