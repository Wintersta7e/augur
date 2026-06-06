"""Descriptor parenthetical renders in single-domain builders + correlation path."""

from unittest.mock import MagicMock

from reasoning.augur_advisor import (
    build_activity_focus_prompt,
    build_activity_intensity_prompt,
    build_correlation_prompt,
)

_FOCUS = {
    "entity": "alpha_app",
    "baseline_mean": 2.0,
    "deviation_score": 3.1,
    "severity": "medium",
    "context": {
        "new_app": "beta_app",
        "active_dwell_s": 42.0,
        "idle_dwell_s": 1.0,
        "total_dwell_s": 43.0,
    },
}
_INTENSITY = {
    "entity": "alpha_app",
    "value": 140.0,
    "baseline_mean": 60.0,
    "deviation_score": 3.0,
    "severity": "medium",
    "context": {
        "keystroke_count": 5,
        "mouse_event_count": 120,
        "idle_seconds": 0.0,
        "window_duration_s": 10.0,
    },
}


def test_focus_prompt_renders_descriptor_when_present():
    a = {**_FOCUS, "context": {**_FOCUS["context"], "app_descriptor": "Alpha Browser"}}
    out = build_activity_focus_prompt(a, MagicMock(), "SYS")
    assert "alpha_app (Alpha Browser)" in out


def test_focus_prompt_omits_descriptor_when_absent():
    out = build_activity_focus_prompt(_FOCUS, MagicMock(), "SYS")
    assert "alpha_app" in out
    assert "alpha_app (" not in out


def test_intensity_prompt_renders_descriptor():
    a = {
        **_INTENSITY,
        "context": {**_INTENSITY["context"], "app_descriptor": "Alpha Browser"},
    }
    out = build_activity_intensity_prompt(a, MagicMock(), "SYS")
    assert "alpha_app (Alpha Browser)" in out


def test_correlation_prompt_renders_activity_descriptor():
    payload = {
        "primary_anomaly": {
            "domain": "activity_intensity",
            **_INTENSITY,
            "context": {**_INTENSITY["context"], "app_descriptor": "Alpha Browser"},
        },
        "correlated_events": [
            {
                "domain": "typing",
                "value": 5.0,
                "unit": "seconds",
                "deviation_score": 2.0,
                "context": {},
            }
        ],
        "temporal_lag_seconds": 3.0,
        "combined_severity": "HIGH",
        "escalation_rule": "",
    }
    out = build_correlation_prompt(payload)
    assert "alpha_app (Alpha Browser)" in out


def test_correlation_prompt_renders_correlated_activity_descriptor():
    payload = {
        "primary_anomaly": {
            "domain": "typing",
            "value": 5.0,
            "unit": "seconds",
            "deviation_score": 2.0,
            "baseline_mean": 1.0,
            "context": {},
        },
        "correlated_events": [
            {
                "domain": "activity_intensity",
                "entity": "alpha_app",
                "value": 140.0,
                "baseline_mean": 60.0,
                "deviation_score": 3.0,
                "context": {
                    "keystroke_count": 5,
                    "app_descriptor": "Alpha Browser",
                },
            }
        ],
        "temporal_lag_seconds": 3.0,
        "combined_severity": "HIGH",
        "escalation_rule": "",
    }
    out = build_correlation_prompt(payload)
    assert "alpha_app (Alpha Browser)" in out
