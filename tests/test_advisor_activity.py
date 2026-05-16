"""Unit tests for the activity_focus / activity_intensity advisor handlers."""

from __future__ import annotations

from unittest.mock import MagicMock

from reasoning.augur_advisor import (
    DOMAIN_HANDLERS,
    build_activity_focus_prompt,
    build_activity_intensity_prompt,
    describe_signal,
)


def test_domain_handlers_registered():
    assert DOMAIN_HANDLERS["activity_focus"] is build_activity_focus_prompt
    assert DOMAIN_HANDLERS["activity_intensity"] is build_activity_intensity_prompt


def test_describe_signal_activity_focus():
    anomaly = {
        "domain": "activity_focus",
        "entity": "code",
        "value": 4.8,  # log1p_seconds
        "unit": "log1p_seconds",
        "context": {"active_dwell_s": 120.0, "new_app": "chrome"},
        "baseline_mean": 3.2,
        "deviation_score": 2.5,
    }
    line = describe_signal("activity_focus", anomaly)
    assert "ACTIVITY_FOCUS" in line
    assert "code" in line
    assert "chrome" in line  # the app you switched TO is informative
    assert "2.5" in line  # deviation


def test_describe_signal_activity_intensity():
    anomaly = {
        "domain": "activity_intensity",
        "entity": "code",
        "value": 320.0,
        "unit": "ipm",
        "context": {"keystroke_count": 50, "mouse_event_count": 4},
        "baseline_mean": 60.0,
        "deviation_score": 5.4,
    }
    line = describe_signal("activity_intensity", anomaly)
    assert "ACTIVITY_INTENSITY" in line
    assert "code" in line
    assert "320" in line  # the ipm value
    assert "5.4" in line


def test_build_activity_focus_prompt_includes_system_prompt_and_fields():
    anomaly = {
        "domain": "activity_focus",
        "entity": "code",
        "value": 4.8,
        "unit": "log1p_seconds",
        "context": {
            "active_dwell_s": 120.0,
            "idle_dwell_s": 30.0,
            "total_dwell_s": 150.0,
            "new_app": "chrome",
            "prev_title": None,
            "new_title": None,
        },
        "baseline_mean": 3.2,
        "deviation_score": 2.5,
        "severity": "HIGH",
    }
    prompt = build_activity_focus_prompt(anomaly, MagicMock(), "You are Augur.")
    assert prompt.startswith("You are Augur.")
    assert "code" in prompt
    assert "chrome" in prompt
    assert "120" in prompt  # active dwell seconds
    assert "HIGH" in prompt
    # No raw-title leakage when allowlist denied them.
    assert "None" in prompt or "title" not in prompt.lower()


def test_build_activity_intensity_prompt_includes_relevant_fields():
    anomaly = {
        "domain": "activity_intensity",
        "entity": "code",
        "value": 320.0,
        "unit": "ipm",
        "context": {
            "keystroke_count": 50,
            "mouse_event_count": 4,
            "idle_seconds": 0.5,
            "window_duration_s": 10.0,
        },
        "baseline_mean": 60.0,
        "deviation_score": 5.4,
        "severity": "HIGH",
    }
    prompt = build_activity_intensity_prompt(anomaly, MagicMock(), "You are Augur.")
    assert prompt.startswith("You are Augur.")
    assert "code" in prompt
    assert "320" in prompt
    assert "60" in prompt  # baseline
    assert "HIGH" in prompt
