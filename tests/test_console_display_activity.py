"""Unit tests for console_display renderers on activity domains."""

from __future__ import annotations

from vox.console_display import render_advice, render_anomaly_line


def test_render_anomaly_line_activity_focus():
    data = {
        "domain": "activity_focus",
        "entity": "code",
        "value": 4.8,
        "unit": "log1p_seconds",
        "baseline_mean": 3.2,
        "deviation_score": 2.5,
        "severity": "medium",
        "context": {
            "active_dwell_s": 120.0,
            "new_app": "chrome",
        },
    }
    line = render_anomaly_line(data)
    assert "activity_focus" in line.lower() or "ACTIVITY_FOCUS" in line
    assert "code" in line
    assert "MEDIUM" in line
    # active dwell, not the log1p value, is what's human-meaningful
    assert "120" in line


def test_render_anomaly_line_activity_intensity():
    data = {
        "domain": "activity_intensity",
        "entity": "code",
        "value": 320.0,
        "unit": "ipm",
        "baseline_mean": 60.0,
        "deviation_score": 5.4,
        "severity": "high",
        "context": {
            "keystroke_count": 50,
            "mouse_event_count": 4,
            "window_duration_s": 10.0,
        },
    }
    line = render_anomaly_line(data)
    assert "activity_intensity" in line.lower() or "ACTIVITY_INTENSITY" in line
    assert "code" in line
    assert "HIGH" in line
    assert "320" in line


def test_render_advice_for_activity_domain_does_not_crash():
    data = {
        "domain": "activity_focus",
        "entity": "code",
        "severity": "medium",
        "advice": "You've been in code for a while. Consider a short break.",
        "correlation_found": False,
    }
    out = render_advice(data)
    assert "code" in out
    assert "break" in out
