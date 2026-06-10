"""Unit tests for the activity_focus / activity_intensity advisor handlers."""

from __future__ import annotations

from unittest.mock import MagicMock

from consilium.advisor import (
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


def test_build_activity_focus_prompt_does_not_leak_unfiltered_title():
    """If a title sneaks into context (e.g., allowlist applied at wrong layer),
    the prompt builder still doesn't expose it via the default templates.

    Sentinel: a unique string only used here. If it appears in the prompt,
    the privacy contract is broken.
    """
    sentinel = "SENTINEL_LEAK_CHECK_ZXQW_8675309"
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
            "prev_title": sentinel,  # would-be leak; daemon-side filter should have removed it
            "new_title": sentinel,
        },
        "baseline_mean": 3.2,
        "deviation_score": 2.5,
        "severity": "HIGH",
    }
    prompt = build_activity_focus_prompt(anomaly, MagicMock(), "You are Augur.")
    assert sentinel not in prompt, (
        f"Sentinel leaked into prompt — privacy contract broken. Got: {prompt!r}"
    )


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


def test_build_activity_intensity_prompt_does_not_default_to_pathological_framing():
    """The advice prompt must not bias the LLM by listing only pathological
    interpretations. Normal heavy use of a click-heavy app (Steam store, video,
    gaming) is a valid interpretation and must be reachable from the prompt.
    """
    anomaly = {
        "domain": "activity_intensity",
        "entity": "clicky-app",
        "value": 433.0,
        "unit": "ipm",
        "context": {
            "keystroke_count": 0,
            "mouse_event_count": 60,
            "idle_seconds": 0.0,
            "window_duration_s": 10.0,
        },
        "baseline_mean": 226.0,
        "deviation_score": 2.7,
        "severity": "MEDIUM",
    }
    prompt = build_activity_intensity_prompt(anomaly, MagicMock(), "You are Augur.")

    biased = "high-energy work, fatigue, automation, or distraction"
    assert biased not in prompt, (
        f"Prompt still uses the biased menu phrasing {biased!r}"
    )

    prompt_lower = prompt.lower()
    assert any(
        phrase in prompt_lower
        for phrase in ("normal heavy use", "normal use", "normal browsing")
    ), "Prompt does not offer a non-pathological framing for high intensity"


def test_build_activity_intensity_prompt_includes_keystroke_click_ratio_guidance():
    """The prompt must explicitly invite the LLM to weigh the keystroke-vs-click
    ratio, because 'ipm' alone conflates typing-heavy and scroll/click-heavy use.
    """
    anomaly = {
        "domain": "activity_intensity",
        "entity": "clicky-app",
        "value": 433.0,
        "unit": "ipm",
        "context": {
            "keystroke_count": 0,
            "mouse_event_count": 60,
            "idle_seconds": 0.0,
            "window_duration_s": 10.0,
        },
        "baseline_mean": 226.0,
        "deviation_score": 2.7,
        "severity": "MEDIUM",
    }
    prompt = build_activity_intensity_prompt(anomaly, MagicMock(), "You are Augur.")
    prompt_lower = prompt.lower()
    assert any(
        phrase in prompt_lower
        for phrase in (
            "keystroke/click ratio",
            "keystroke and click",
            "keystrokes and clicks",
            "typing vs",
            "typing-heavy",
            "click-heavy",
        )
    ), "Prompt does not guide the LLM to consider the keystroke/click breakdown"


def test_build_activity_focus_prompt_does_not_default_to_pathological_framing():
    """Focus prompt currently asks 'deep focus vs stuck vs distraction' — 2/3 negative.
    Normal/varied task duration must be reachable from the prompt.
    """
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

    biased = "deep focus vs stuck vs distraction"
    assert biased not in prompt, (
        f"Focus prompt still uses biased menu phrasing {biased!r}"
    )

    prompt_lower = prompt.lower()
    assert any(
        phrase in prompt_lower
        for phrase in (
            "normal task duration",
            "varied task",
            "ordinary task",
            "routine task",
        )
    ), "Focus prompt does not offer a non-pathological framing"


def test_domain_handlers_and_describers_keys_match():
    """Each domain with a prompt builder must also have a describer (and vice versa)."""
    from consilium.advisor import DOMAIN_HANDLERS, DOMAIN_DESCRIBERS

    assert set(DOMAIN_HANDLERS.keys()) == set(DOMAIN_DESCRIBERS.keys()), (
        f"DOMAIN_HANDLERS keys {set(DOMAIN_HANDLERS.keys())} "
        f"don't match DOMAIN_DESCRIBERS keys {set(DOMAIN_DESCRIBERS.keys())}"
    )
