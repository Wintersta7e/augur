"""Tests for classify_severity — the gate that controls LLM activation.

If this function misclassifies severity, either the LLM fires on noise
(wasting Ollama calls) or misses genuine anomalies (defeating the purpose).
"""

from __future__ import annotations

import pytest

from detection.anomaly_detector import classify_severity, DEFAULT_THRESHOLDS

# Use the actual project defaults for boundary tests
MEDIUM = DEFAULT_THRESHOLDS["severity_medium_sigma"]  # 2.5
HIGH = DEFAULT_THRESHOLDS["severity_high_sigma"]      # 4.0


class TestSeverityBoundaries:
    """Verify exact boundary behavior with default thresholds."""

    def test_below_medium_is_low(self) -> None:
        assert classify_severity(2.0, 0.5, MEDIUM, HIGH) == "low"

    def test_at_medium_sigma_is_medium(self) -> None:
        assert classify_severity(MEDIUM, 0.5, MEDIUM, HIGH) == "medium"

    def test_between_medium_and_high_is_medium(self) -> None:
        assert classify_severity(3.0, 0.5, MEDIUM, HIGH) == "medium"

    def test_at_high_sigma_is_high(self) -> None:
        assert classify_severity(HIGH, 0.5, MEDIUM, HIGH) == "high"

    def test_above_high_sigma_is_high(self) -> None:
        assert classify_severity(10.0, 0.5, MEDIUM, HIGH) == "high"


class TestHSTOverrides:
    """HST score can override sigma-based classification."""

    def test_hst_09_forces_high_regardless_of_sigma(self) -> None:
        assert classify_severity(1.0, 0.9, MEDIUM, HIGH) == "high"

    def test_hst_08_forces_medium_regardless_of_sigma(self) -> None:
        assert classify_severity(1.0, 0.8, MEDIUM, HIGH) == "medium"

    def test_hst_below_08_does_not_override(self) -> None:
        assert classify_severity(1.0, 0.79, MEDIUM, HIGH) == "low"


class TestEdgeCases:
    """Guard against degenerate inputs."""

    def test_zero_deviation_zero_hst(self) -> None:
        assert classify_severity(0.0, 0.0, MEDIUM, HIGH) == "low"

    def test_negative_deviation_treated_as_low(self) -> None:
        # Shouldn't happen (abs value), but classify_severity should not crash
        assert classify_severity(-5.0, 0.0, MEDIUM, HIGH) == "low"

    def test_hst_exactly_one(self) -> None:
        assert classify_severity(0.0, 1.0, MEDIUM, HIGH) == "high"
