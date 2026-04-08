"""Tests for analyze_precision — sigma threshold self-adjustment.

This is the function that decides whether Augur should become more or less
sensitive. A bug here means Augur tunes itself in the wrong direction:
raising thresholds when it should lower them, or vice versa. The system
would confidently degrade while reporting that it adjusted itself.
"""

from __future__ import annotations

import pytest

from blackboard.config import AugurConfig
from reasoning.reflection_engine import analyze_precision

_CFG = AugurConfig()
SIGMA_ADJUST_STEP = _CFG.sigma_adjust_step
SIGMA_MIN = _CFG.sigma_min
SIGMA_MAX = _CFG.sigma_max


def _make_feedback(
    ratings: list[str],
    behavioral_scores: list[float] | None = None,
) -> dict:
    """Build a minimal feedback dict with the given explicit ratings."""
    events = []
    for i, rating in enumerate(ratings):
        ev = {"explicit_rating": rating}
        if behavioral_scores and i < len(behavioral_scores):
            ev["behavioral_score"] = behavioral_scores[i]
        events.append(ev)
    return {
        "advice_events": events,
        "session_summary": {"total_advice": len(events)},
    }


class TestPrecisionNoAnomalies:
    """When no anomalies fired, nothing should change."""

    def test_empty_session_returns_no_action(self) -> None:
        result = analyze_precision(
            {"advice_events": [], "session_summary": {"total_advice": 0}},
            {"sigma_threshold": 2.0},
            _CFG,
        )
        assert result["action"] == "none"
        assert result["precision_ratio"] == 1.0
        assert result["sigma_before"] == result["sigma_after"]


class TestPrecisionLow:
    """Low precision (< 0.3) should raise sigma to reduce noise."""

    def test_all_negative_raises_sigma(self) -> None:
        feedback = _make_feedback(["n", "n", "n"])
        result = analyze_precision(feedback, {"sigma_threshold": 2.0}, _CFG)
        assert result["action"] == "raise_sigma"
        assert result["sigma_after"] == pytest.approx(2.0 + SIGMA_ADJUST_STEP)

    def test_low_precision_with_behavioral_fallback(self) -> None:
        # All explicit negative, but one has high behavioral score
        feedback = _make_feedback(["n", "n", "n"], [0.8, 0.1, 0.1])
        result = analyze_precision(feedback, {"sigma_threshold": 2.0}, _CFG)
        # 1 useful out of 3 = 0.33, just above 0.3 threshold
        assert result["precision_ratio"] == pytest.approx(1 / 3, abs=0.01)
        assert result["action"] == "none"  # 0.33 > 0.3, no action

    def test_sigma_capped_at_max(self) -> None:
        feedback = _make_feedback(["n", "n"])
        result = analyze_precision(feedback, {"sigma_threshold": SIGMA_MAX}, _CFG)
        assert result["sigma_after"] == SIGMA_MAX


class TestPrecisionHigh:
    """High precision (> 0.8) should lower sigma to catch more."""

    def test_all_positive_lowers_sigma(self) -> None:
        feedback = _make_feedback(["y", "y", "y"])
        result = analyze_precision(feedback, {"sigma_threshold": 2.5}, _CFG)
        assert result["action"] == "lower_sigma"
        assert result["sigma_after"] == pytest.approx(2.5 - SIGMA_ADJUST_STEP)

    def test_sigma_floored_at_min(self) -> None:
        feedback = _make_feedback(["y", "y"])
        result = analyze_precision(feedback, {"sigma_threshold": SIGMA_MIN}, _CFG)
        assert result["sigma_after"] == SIGMA_MIN


class TestPrecisionMinimumSampleSize:
    """Need at least 2 anomalies to adjust — single-event sessions are noise."""

    def test_single_negative_no_action(self) -> None:
        feedback = _make_feedback(["n"])
        result = analyze_precision(feedback, {"sigma_threshold": 2.0}, _CFG)
        assert result["action"] == "none"

    def test_single_positive_no_action(self) -> None:
        feedback = _make_feedback(["y"])
        result = analyze_precision(feedback, {"sigma_threshold": 2.0}, _CFG)
        assert result["action"] == "none"


class TestPrecisionMiddleRange:
    """Precision between 0.3 and 0.8 should not adjust."""

    def test_half_positive_no_action(self) -> None:
        feedback = _make_feedback(["y", "n", "y", "n"])
        result = analyze_precision(feedback, {"sigma_threshold": 2.0}, _CFG)
        assert result["precision_ratio"] == pytest.approx(0.5)
        assert result["action"] == "none"
